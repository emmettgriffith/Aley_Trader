"""Footprint chart utilities for Aley Trader.

This module adapts the provided footprint template so it can be reused inside
the application.  Ticks are aggregated into bars, binned by price, and rendered
with a dark theme that matches the rest of the UI.

Usage
-----

    from footprint_chart import render_footprint
    fig = render_footprint(ticks_df, tick_size=0.25, interval="1min")

The returned matplotlib Figure can be embedded into the Tk canvas just like
other chart types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import math
import queue
import threading
import time

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Theme – roughly matches DEEP_SEA_THEME from aley_trader.py.
# ---------------------------------------------------------------------------
BACKGROUND = "#0A1628"
TEXT_COLOR = "#ECF0F1"
GRID_COLOR = "#253244"
ASK_COLOR = "#27AE60"  # buyers/ask aggressors
BID_COLOR = "#E74C3C"  # sellers/bid aggressors
VWAP_COLOR = "#5DADE2"
PROFILE_COLOR = "#2C3E50"

DEFAULT_INTERVAL = "1min"
DEFAULT_FONT_SIZE = 8


# ---------------------------------------------------------------------------
# Helpers for trade side inference and tick rounding
# ---------------------------------------------------------------------------
def infer_side_from_last_price(ticks: pd.DataFrame) -> pd.Series:
    """Infer aggressor side when the feed does not provide one.

    The first trade is assumed to be an ask. For subsequent trades we compare
    the price with the previous trade: upticks are treated as ask-aggressor
    (buyers hitting offers) while downticks are treated as bid-aggressor.
    Flat trades reuse the last known side.
    """

    prices = ticks["price"].to_numpy()
    sides = np.empty(len(prices), dtype=object)
    last_side = "ASK"
    for idx, price in enumerate(prices):
        if idx == 0:
            sides[idx] = last_side
            continue
        prev_price = prices[idx - 1]
        if price > prev_price:
            last_side = "ASK"
        elif price < prev_price:
            last_side = "BID"
        sides[idx] = last_side
    return pd.Series(sides, index=ticks.index, name="side")


def _round_to_tick(price: float, tick_size: float) -> float:
    return round(round(price / tick_size) * tick_size, 10)


# ---------------------------------------------------------------------------
# Data classes for footprint bars
# ---------------------------------------------------------------------------
@dataclass
class Cell:
    price: float
    bid_vol: float
    ask_vol: float


@dataclass
class FootprintBar:
    start: pd.Timestamp
    end: pd.Timestamp
    vwap: float
    high: float
    low: float
    total_volume: float
    cells: List[Cell]


# ---------------------------------------------------------------------------
# Aggregation logic – converts ticks into FootprintBar objects
# ---------------------------------------------------------------------------
def build_footprints(
    ticks: pd.DataFrame,
    *,
    interval: str = DEFAULT_INTERVAL,
    tick_size: float,
) -> List[FootprintBar]:
    """Aggregate raw ticks into footprint bars.

    Parameters
    ----------
    ticks : DataFrame
        Columns required: ``ts`` (datetime64), ``price`` (float), ``size``
        (float). Optional ``side`` column with 'BID'/'ASK'.
    interval : str
        Pandas offset alias used to bucket timestamps (e.g. '15S', '1min').
    tick_size : float
        Minimum price increment for the symbol.
    """

    if "ts" not in ticks.columns:
        raise ValueError("ticks must include a 'ts' timestamp column")
    if "price" not in ticks.columns or "size" not in ticks.columns:
        raise ValueError("ticks must include 'price' and 'size' columns")

    df = ticks.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        raise ValueError("ticks['ts'] must be datetime64")

    df = df.sort_values("ts")
    if "side" in df.columns:
        side = df["side"].astype(str).str.upper().fillna("")
        side = side.where(side.isin(["BID", "ASK"]), None)
    else:
        side = None

    if side is None or side.isna().any():
        side = infer_side_from_last_price(df)
    df["side"] = side

    df["bar_ts"] = df["ts"].dt.floor(interval)
    df["tick_index"] = (df["price"] / tick_size).round().astype(int)

    bars: List[FootprintBar] = []
    for bar_start, chunk in df.groupby("bar_ts"):
        if chunk.empty:
            continue

        bar_start = pd.Timestamp(bar_start)
        bar_end = bar_start + pd.tseries.frequencies.to_offset(interval)
        high = float(chunk["price"].max())
        low = float(chunk["price"].min())
        vol = float(chunk["size"].sum())
        weights = chunk["size"].to_numpy()
        prices = chunk["price"].to_numpy()
        vwap = float(np.average(prices, weights=weights)) if weights.sum() else float(np.mean(prices))

        price_bins: Dict[int, Tuple[float, float]] = {}
        for tick_idx, subset in chunk.groupby("tick_index"):
            bid_vol = float(subset.loc[subset["side"] == "BID", "size"].sum())
            ask_vol = float(subset.loc[subset["side"] == "ASK", "size"].sum())
            price_bins[tick_idx] = (bid_vol, ask_vol)

        tick_min = int(math.floor(low / tick_size))
        tick_max = int(math.ceil(high / tick_size))
        cells: List[Cell] = []
        for tick_idx in range(tick_max, tick_min - 1, -1):
            bid_vol, ask_vol = price_bins.get(tick_idx, (0.0, 0.0))
            price = _round_to_tick(tick_idx * tick_size, tick_size)
            cells.append(Cell(price=price, bid_vol=bid_vol, ask_vol=ask_vol))

        bars.append(
            FootprintBar(
                start=bar_start,
                end=bar_end,
                vwap=vwap,
                high=high,
                low=low,
                total_volume=vol,
                cells=cells,
            )
        )

    return bars


# ---------------------------------------------------------------------------
# Rendering logic
# ---------------------------------------------------------------------------
def plot_footprints(
    bars: Iterable[FootprintBar],
    *,
    show_volume_profile: bool = True,
    max_bars: Optional[int] = 40,
    figsize: Tuple[int, int] = (16, 9),
    title: Optional[str] = None,
) -> Figure:
    """Create the footprint figure for the provided bars."""

    bars = list(bars)
    if max_bars is not None:
        bars = bars[-max_bars:]
    if not bars:
        raise ValueError("No bars available for plotting")

    all_prices = [cell.price for bar in bars for cell in bar.cells]
    unique_prices = sorted(set(all_prices))
    if len(unique_prices) > 1:
        tick_size = min(np.diff(unique_prices))
    else:
        tick_size = 1.0

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.06, 0.08, 0.86, 0.85])
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    bar_gap = 0.55
    bar_width = 1.35
    positions = np.arange(len(bars)) * (bar_width + bar_gap)

    volume_profile: Dict[float, float] = {}

    for idx, bar in enumerate(bars):
        x0 = positions[idx]
        for cell in bar.cells:
            y = cell.price
            height = tick_size
            rect = plt.Rectangle(
                (x0, y - height / 2),
                bar_width,
                height,
                facecolor="#1B2838",
                edgecolor=GRID_COLOR,
                linewidth=0.6,
            )
            ax.add_patch(rect)

            sell_txt = f"{int(cell.bid_vol)}"
            buy_txt = f"{int(cell.ask_vol)}"
            mid_y = y
            ax.text(
                x0 + bar_width * 0.25,
                mid_y,
                sell_txt,
                ha="right",
                va="center",
                fontsize=DEFAULT_FONT_SIZE,
                color=BID_COLOR,
                family="DejaVu Sans Mono",
            )
            ax.text(
                x0 + bar_width * 0.75,
                mid_y,
                buy_txt,
                ha="left",
                va="center",
                fontsize=DEFAULT_FONT_SIZE,
                color=ASK_COLOR,
                family="DejaVu Sans Mono",
            )

            volume_profile[y] = volume_profile.get(y, 0.0) + cell.bid_vol + cell.ask_vol

        ax.plot(
            [x0, x0 + bar_width],
            [bar.vwap, bar.vwap],
            color=VWAP_COLOR,
            linewidth=1.0,
            alpha=0.8,
        )
        ax.text(
            x0 + bar_width / 2,
            bar.high + tick_size,
            bar.start.strftime("%H:%M"),
            ha="center",
            va="bottom",
            fontsize=DEFAULT_FONT_SIZE,
            color=TEXT_COLOR,
        )

    if show_volume_profile and volume_profile:
        vp_ax = fig.add_axes([0.93, 0.08, 0.05, 0.85], sharey=ax)
        vp_ax.set_facecolor(BACKGROUND)
        max_vol = max(volume_profile.values())
        for price, vol in volume_profile.items():
            width = vol / max_vol
            vp_ax.barh(
                price,
                width,
                height=tick_size * 0.8,
                color=PROFILE_COLOR,
                alpha=0.6,
            )
        vp_ax.invert_xaxis()
        vp_ax.tick_params(axis='x', colors=TEXT_COLOR, labelsize=7)
        vp_ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=7)
        vp_ax.spines['top'].set_visible(False)
        vp_ax.spines['right'].set_visible(False)
        vp_ax.spines['left'].set_visible(False)
        vp_ax.spines['bottom'].set_color(GRID_COLOR)
        vp_ax.set_xlabel("Vol", color=TEXT_COLOR, fontsize=8)
    else:
        vp_ax = None

    y_min = min(all_prices) - tick_size * 3
    y_max = max(all_prices) + tick_size * 3

    if len(positions) > 0:
        x_min = positions[0] - bar_gap
        x_max = positions[-1] + bar_width + bar_gap
    else:
        x_min, x_max = -1, 1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.tick_params(axis='x', colors=TEXT_COLOR, labelsize=8)
    ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=8)

    if tick_size <= 0:
        decimals = 2
    else:
        decimals = max(0, min(6, int(round(-math.log10(tick_size))) if tick_size < 1 else 0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{decimals}f}"))
    ax.set_xlabel("Time", color=TEXT_COLOR)
    ax.set_ylabel("Price", color=TEXT_COLOR)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=13, pad=10)

    return fig


def render_footprint(
    ticks: pd.DataFrame,
    *,
    tick_size: float,
    interval: str = DEFAULT_INTERVAL,
    show_volume_profile: bool = True,
    max_bars: Optional[int] = 40,
    title: Optional[str] = None,
) -> Figure:
    """Public helper that aggregates ticks then renders the footprint figure."""

    bars = build_footprints(ticks, interval=interval, tick_size=tick_size)
    if title is None and not ticks.empty:
        title = f"Footprint ({interval})"
    fig = plot_footprints(
        bars,
        show_volume_profile=show_volume_profile,
        max_bars=max_bars,
        title=title,
    )
    return fig


class FootprintRenderWorker:
    """Background worker that batches ticks and renders footprints without
    blocking the UI thread."""

    def __init__(
        self,
        *,
        tick_size: float,
        interval: str,
        show_profile_fn: Callable[[], bool],
        update_callback: Callable[[Figure], None],
        batch_ms: int = 400,
        max_ticks: int = 10_000,
        max_bars: int = 40,
    ) -> None:
        self.tick_size = tick_size
        self.interval = interval
        self.show_profile_fn = show_profile_fn
        self.update_callback = update_callback
        self.batch_ms = max(100, batch_ms)
        self.max_ticks = max(1000, max_ticks)
        self.max_bars = max_bars

        self._queue: "queue.Queue[pd.DataFrame]" = queue.Queue()
        self._buffer = pd.DataFrame(columns=["ts", "price", "size", "side"])
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- configuration -----------------------------------------------------
    def set_interval(self, interval: str) -> None:
        self.interval = interval

    def set_tick_size(self, tick_size: float) -> None:
        self.tick_size = tick_size

    def set_max_bars(self, max_bars: int) -> None:
        self.max_bars = max_bars

    def clear(self) -> None:
        with self._lock:
            self._buffer = self._buffer.iloc[0:0]

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="FootprintRenderWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.5)
        self._thread = None

    # -- input -------------------------------------------------------------
    def submit(self, ticks: pd.DataFrame) -> None:
        if ticks is None or ticks.empty:
            return
        required = {"ts", "price", "size"}
        if not required.issubset(ticks.columns):
            raise ValueError("ticks must include ts, price, and size columns")
        self._queue.put(ticks.copy())

    # -- worker loop -------------------------------------------------------
    def _run(self) -> None:
        flush_seconds = self.batch_ms / 1000.0
        while not self._stop_event.is_set():
            try:
                first = self._queue.get(timeout=flush_seconds)
            except queue.Empty:
                continue

            batches = [first]
            start_time = time.monotonic()
            while (time.monotonic() - start_time) < flush_seconds:
                try:
                    batches.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            combined = pd.concat(batches, ignore_index=True)
            combined["ts"] = pd.to_datetime(combined["ts"], utc=True, errors="coerce")
            combined = combined.dropna(subset=["ts", "price", "size"])
            if "side" not in combined.columns:
                combined["side"] = np.nan

            with self._lock:
                self._buffer = pd.concat([self._buffer, combined], ignore_index=True)
                if len(self._buffer) > self.max_ticks:
                    self._buffer = self._buffer.iloc[-self.max_ticks :].reset_index(drop=True)
                buffer_copy = self._buffer.copy()

            try:
                fig = render_footprint(
                    buffer_copy,
                
                    tick_size=self.tick_size,
                    interval=self.interval,
                    show_volume_profile=self.show_profile_fn(),
                    max_bars=self.max_bars,
                    title=f"Footprint ({self.interval})",
                )
            except Exception:
                continue

            try:
                self.update_callback(fig)
            except Exception:
                plt.close(fig)


__all__ = [
    "Cell",
    "FootprintBar",
    "FootprintRenderWorker",
    "build_footprints",
    "plot_footprints",
    "render_footprint",
    "infer_side_from_last_price",
]
