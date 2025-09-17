"""overflow_chart.py
Professional overflow chart utilities (Matplotlib + Plotly).

Quick start:
    from overflow_chart import plot_overflow_matplotlib, plot_overflow_plotly
    fig = plot_overflow_matplotlib(df, x="time", y="value", threshold=100)
    fig.show()  # or plt.show()

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - seaborn  (optional, for themes)
    - plotly    (optional, for interactive)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Optional styling
try:
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

# Optional interactive backend
try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


def _validate_inputs(
    df: pd.DataFrame,
    x: str,
    y: str,
    threshold: float | int
) -> None:
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Columns '{x}' and/or '{y}' not in dataframe: {df.columns.tolist()}")
    if not np.isscalar(threshold):
        raise ValueError("'threshold' must be a scalar (int or float).")


def compute_overflow_mask(series: pd.Series, threshold: float | int) -> pd.Series:
    """Return a boolean Series where values exceed the threshold."""
    return series > threshold


def overflow_stats(series: pd.Series, threshold: float | int) -> dict:
    """Compute simple overflow statistics for a numeric Series.

    Returns a dict with:
        - count_over: number of points above threshold
        - pct_over: percentage of points above threshold
        - max_over: maximum value above threshold (nan if none)
        - mean_over: mean value of points above threshold (nan if none)
        - total_excess: sum of (value - threshold) over overflow points
    """
    mask = compute_overflow_mask(series, threshold)
    over = series[mask]
    if over.empty:
        return {
            "count_over": 0,
            "pct_over": 0.0,
            "max_over": float("nan"),
            "mean_over": float("nan"),
            "total_excess": 0.0,
        }
    excess = (over - threshold).clip(lower=0)
    return {
        "count_over": int(mask.sum()),
        "pct_over": float(mask.mean() * 100.0),
        "max_over": float(over.max()),
        "mean_over": float(over.mean()),
        "total_excess": float(excess.sum()),
    }


def plot_volume_profile_chart(
    df: pd.DataFrame,
    price_col: str = "Close",
    volume_col: str = "Volume",
    bins: int = 50,
    *,
    title: str | None = "Volume Profile Chart",
    figsize: tuple[int, int] = (14, 10),
    show_poc: bool = True,
    show_vah_val: bool = True
):
    """
    Create a volume profile chart showing price-volume relationships
    similar to order book depth visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price and volume data
    price_col : str
        Column name for price data
    volume_col : str
        Column name for volume data
    bins : int
        Number of price bins for volume profile
    title : str
        Chart title
    figsize : tuple
        Figure size (width, height)
    show_poc : bool
        Show Point of Control (highest volume price)
    show_vah_val : bool
        Show Value Area High/Low (70% volume area)
    
    Returns:
    --------
    tuple : (figure, axes, profile_stats)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Validate inputs
    if df.empty:
        raise ValueError("Data cannot be empty")
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in data")
    if volume_col not in df.columns:
        raise ValueError(f"Volume column '{volume_col}' not found in data")
    
    # Set up the figure with custom layout
    fig = plt.figure(figsize=figsize, facecolor='#0A1628')
    
    # Create custom grid layout
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    
    # Main price chart (left)
    ax_main = fig.add_subplot(gs[0])
    ax_main.set_facecolor('#0A1628')
    
    # Volume profile (right)
    ax_profile = fig.add_subplot(gs[1], sharey=ax_main)
    ax_profile.set_facecolor('#0A1628')
    
    # Calculate price range and create bins
    price_data = df[price_col].dropna()
    volume_data = df[volume_col].dropna()
    
    min_price = price_data.min()
    max_price = price_data.max()
    price_bins = np.linspace(min_price, max_price, bins + 1)
    
    # Create volume profile by binning prices
    volume_profile = []
    bin_centers = []
    
    for i in range(len(price_bins) - 1):
        bin_min = price_bins[i]
        bin_max = price_bins[i + 1]
        bin_center = (bin_min + bin_max) / 2
        
        # Find data points in this price range
        in_bin = (price_data >= bin_min) & (price_data < bin_max)
        bin_volume = volume_data[in_bin].sum()
        
        volume_profile.append(bin_volume)
        bin_centers.append(bin_center)
    
    volume_profile = np.array(volume_profile)
    bin_centers = np.array(bin_centers)
    
    # Calculate key levels
    total_volume = volume_profile.sum()
    
    # Point of Control (POC) - price level with highest volume
    poc_idx = np.argmax(volume_profile)
    poc_price = bin_centers[poc_idx]
    poc_volume = volume_profile[poc_idx]
    
    # Value Area High/Low (VAH/VAL) - 70% of volume
    sorted_indices = np.argsort(volume_profile)[::-1]  # Sort by volume descending
    cumulative_volume = 0
    value_area_indices = []
    
    for idx in sorted_indices:
        cumulative_volume += volume_profile[idx]
        value_area_indices.append(idx)
        if cumulative_volume >= 0.7 * total_volume:
            break
    
    vah_price = bin_centers[max(value_area_indices)]  # Value Area High
    val_price = bin_centers[min(value_area_indices)]  # Value Area Low
    
    # Plot main price chart with candlestick-like visualization
    if 'Open' in df.columns and 'High' in df.columns and 'Low' in df.columns:
        # Plot candlestick chart
        for i, (idx, row) in enumerate(df.iterrows()):
            if i % max(1, len(df) // 100) == 0:  # Sample for performance
                open_price = row['Open']
                close_price = row[price_col]
                high_price = row['High']
                low_price = row['Low']
                
                color = '#27AE60' if close_price >= open_price else '#E74C3C'
                
                # Plot high-low line
                ax_main.plot([i, i], [low_price, high_price], color=color, alpha=0.8, linewidth=1)
                
                # Plot open-close body
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                ax_main.bar(i, body_height, bottom=body_bottom, width=0.8, 
                           color=color, alpha=0.7, edgecolor=color)
    else:
        # Plot simple line chart
        ax_main.plot(range(len(price_data)), price_data, color='#3498DB', linewidth=2, alpha=0.8)
    
    # Plot volume profile as horizontal bars
    bar_width = (max_price - min_price) / bins * 0.8
    max_volume = volume_profile.max()
    
    # Normalize volume for display
    normalized_volumes = volume_profile / max_volume if max_volume > 0 else volume_profile
    
    for i, (price, volume, norm_vol) in enumerate(zip(bin_centers, volume_profile, normalized_volumes)):
        if volume > 0:
            # Color based on volume intensity
            if norm_vol > 0.8:
                color = '#E74C3C'  # High volume - red
            elif norm_vol > 0.5:
                color = '#F39C12'  # Medium volume - orange
            elif norm_vol > 0.2:
                color = '#3498DB'  # Low-medium volume - blue
            else:
                color = '#7F8C8D'  # Low volume - gray
            
            # Plot horizontal bar
            ax_profile.barh(price, norm_vol, height=bar_width, 
                           color=color, alpha=0.7, edgecolor='none')
    
    # Highlight key levels
    if show_poc:
        # Point of Control
        ax_main.axhline(y=poc_price, color='#F1C40F', linestyle='-', linewidth=2, alpha=0.8, label=f'POC: ${poc_price:.2f}')
        ax_profile.axhline(y=poc_price, color='#F1C40F', linestyle='-', linewidth=2, alpha=0.8)
    
    if show_vah_val:
        # Value Area High/Low
        ax_main.axhline(y=vah_price, color='#9B59B6', linestyle='--', linewidth=1.5, alpha=0.7, label=f'VAH: ${vah_price:.2f}')
        ax_main.axhline(y=val_price, color='#9B59B6', linestyle='--', linewidth=1.5, alpha=0.7, label=f'VAL: ${val_price:.2f}')
        ax_profile.axhline(y=vah_price, color='#9B59B6', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_profile.axhline(y=val_price, color='#9B59B6', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Styling
    ax_main.set_title(title, color='#ECF0F1', fontsize=16, fontweight='bold', pad=20)
    ax_main.set_xlabel('Time Period', color='#BDC3C7', fontsize=12)
    ax_main.set_ylabel('Price ($)', color='#BDC3C7', fontsize=12)
    ax_main.grid(True, alpha=0.3, color='#4A5568')
    ax_main.tick_params(colors='#BDC3C7')
    
    ax_profile.set_xlabel('Volume', color='#BDC3C7', fontsize=12)
    ax_profile.set_title('Volume Profile', color='#ECF0F1', fontsize=12, fontweight='bold')
    ax_profile.grid(True, alpha=0.3, color='#4A5568')
    ax_profile.tick_params(colors='#BDC3C7')
    
    # Remove y-axis labels from profile chart (shared with main chart)
    ax_profile.tick_params(axis='y', labelleft=False)
    
    # Add legend to main chart
    if show_poc or show_vah_val:
        legend = ax_main.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('#2C3E50')
        legend.get_frame().set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color('#ECF0F1')
    
    # Add statistics text box
    stats_text = (
        f"Volume Profile Analysis:\n"
        f"POC: ${poc_price:.2f} ({poc_volume:,.0f} vol)\n"
        f"VAH: ${vah_price:.2f}\n"
        f"VAL: ${val_price:.2f}\n"
        f"Total Volume: {total_volume:,.0f}\n"
        f"Price Range: ${min_price:.2f} - ${max_price:.2f}"
    )
    
    ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes, 
                fontsize=10, verticalalignment='top', color='#ECF0F1',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#2C3E50', alpha=0.8))
    
    plt.tight_layout()
    
    # Calculate return statistics
    profile_stats = {
        'poc_price': poc_price,
        'poc_volume': poc_volume,
        'vah_price': vah_price,
        'val_price': val_price,
        'total_volume': total_volume,
        'value_area_volume': cumulative_volume,
        'price_range': max_price - min_price,
        'volume_profile': volume_profile,
        'price_bins': bin_centers
    }
    
    return fig, (ax_main, ax_profile), profile_stats


def plot_volume_profile_plus(
    df: pd.DataFrame,
    *,
    price_col: str = "Close",
    volume_col: str = "Volume",
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    bins: int = 40,
    title: str | None = None,
    figsize: tuple[int, int] = (14, 9)
):
    """Render a dual-panel chart: price action + buy/sell volume profile."""

    required = {price_col, volume_col, open_col, high_col, low_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for profile: {sorted(missing)}")
    if df.empty:
        raise ValueError("Volume profile data cannot be empty")

    price_min = float(df[low_col].min())
    price_max = float(df[high_col].max())
    if price_min == price_max:
        price_max = price_min + 1e-6

    bins = max(6, int(bins))
    price_bins = np.linspace(price_min, price_max, bins + 1)
    centers = (price_bins[:-1] + price_bins[1:]) / 2
    buy_profile = np.zeros(len(centers))
    sell_profile = np.zeros(len(centers))

    volumes = df[volume_col].to_numpy()
    opens = df[open_col].to_numpy()
    closes = df[price_col].to_numpy()
    highs = df[high_col].to_numpy()
    lows = df[low_col].to_numpy()

    for idx in range(len(centers)):
        upper = price_bins[idx + 1]
        lower = price_bins[idx]
        mask = (lows < upper) & (highs > lower)
        if not np.any(mask):
            continue

        vol_slice = volumes[mask]
        open_slice = opens[mask]
        close_slice = closes[mask]

        buy_mask = close_slice > open_slice
        sell_mask = close_slice < open_slice
        neutral_mask = ~(buy_mask | sell_mask)

        buy_total = vol_slice[buy_mask].sum()
        sell_total = vol_slice[sell_mask].sum()
        neutral_total = vol_slice[neutral_mask].sum()

        buy_profile[idx] = buy_total + neutral_total * 0.5
        sell_profile[idx] = sell_total + neutral_total * 0.5

    total_buy = float(buy_profile.sum())
    total_sell = float(sell_profile.sum())
    total_volume = total_buy + total_sell
    net_delta = total_buy - total_sell
    ratio_pct = (net_delta / total_volume * 100.0) if total_volume else 0.0

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('#0A1628')
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_profile = fig.add_subplot(gs[1], sharey=ax_main)
    ax_main.set_facecolor('#0A1628')
    ax_profile.set_facecolor('#0A1628')

    # Simple candlestick-style plot
    for idx, row in df.reset_index().iterrows():
        open_price = row[open_col]
        close_price = row[price_col]
        high_price = row[high_col]
        low_price = row[low_col]
        if pd.isna(open_price) or pd.isna(close_price):
            continue
        color = '#27AE60' if close_price >= open_price else '#E74C3C'
        ax_main.plot([idx, idx], [low_price, high_price], color=color, linewidth=1)
        ax_main.bar(
            idx,
            height=abs(close_price - open_price),
            bottom=min(open_price, close_price),
            width=0.6,
            color=color,
            alpha=0.8,
            edgecolor=color
        )

    ax_main.set_title(title or "Volume Profile Plus", color='#ECF0F1', fontsize=14)
    ax_main.set_xlabel('Bars', color='#ECF0F1')
    ax_main.set_ylabel('Price', color='#ECF0F1')
    ax_main.tick_params(colors='#BDC3C7')
    ax_main.grid(True, color='#253244', linestyle='--', linewidth=0.3, alpha=0.3)

    max_side = max(float(buy_profile.max()), float(sell_profile.max()), 1.0)
    bar_height = (price_bins[1] - price_bins[0]) * 0.85
    ax_profile.barh(
        centers,
        buy_profile,
        height=bar_height,
        color='#27AE60',
        alpha=0.75,
        label='Buy Volume'
    )
    ax_profile.barh(
        centers,
        -sell_profile,
        height=bar_height,
        color='#E74C3C',
        alpha=0.75,
        label='Sell Volume'
    )
    ax_profile.axvline(0, color='#ECF0F1', linestyle='--', linewidth=1, alpha=0.6)
    ax_profile.set_xlim(-max_side * 1.1, max_side * 1.1)
    ax_profile.set_xlabel('Volume', color='#ECF0F1')
    ax_profile.tick_params(colors='#BDC3C7')
    ax_profile.grid(True, color='#253244', linestyle='--', linewidth=0.3, alpha=0.4)
    ax_profile.legend(facecolor='#1B2838', edgecolor='#4A5568', labelcolor='#ECF0F1')
    ax_profile.tick_params(axis='y', labelleft=False)

    stats_text = (
        f"Total Volume: {total_volume:,.0f}\n"
        f"Buy Volume: {total_buy:,.0f}\n"
        f"Sell Volume: {total_sell:,.0f}\n"
        f"Net Delta: {net_delta:,.0f} ({ratio_pct:+.1f}%)"
    )
    ax_main.text(
        0.02,
        0.98,
        stats_text,
        transform=ax_main.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        color='#ECF0F1',
        bbox=dict(boxstyle='round', facecolor='#1B2838', edgecolor='#4A5568', alpha=0.85)
    )

    stats = {
        'total_volume': total_volume,
        'total_buy_volume': total_buy,
        'total_sell_volume': total_sell,
        'net_delta': net_delta,
        'buy_sell_ratio': ratio_pct,
        'price_bins': price_bins,
        'buy_profile': buy_profile,
        'sell_profile': sell_profile,
    }

    plt.tight_layout()
    return fig, (ax_main, ax_profile), stats


def plot_order_flow_footprint(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    title: str = "Order Flow Footprint",
    figsize: tuple[int, int] = (16, 9),
    max_bars: int = 18,
    levels: int = 12,
    price_precision: int | None = None
):
    """Render a stylised order-flow footprint chart using OHLCV data.

    The footprint approximates bid/ask aggressor volume at each price level by
    blending bar direction, range, and total volume. While it does not consume
    a real order book feed, the visual mirrors professional layouts so traders
    can monitor imbalances and delta behaviour."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if df.empty:
        raise ValueError("Order flow chart requires non-empty price data")

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for order flow chart: {sorted(missing)}")

    plot_df = df.copy()
    if "Date" in plot_df.columns:
        if np.issubdtype(plot_df["Date"].dtype, np.number):
            plot_df["_dt"] = plot_df["Date"].apply(lambda x: mdates.num2date(x).replace(tzinfo=None))
        else:
            plot_df["_dt"] = pd.to_datetime(plot_df["Date"])
    else:
        plot_df["_dt"] = pd.to_datetime(plot_df.index)

    plot_df = plot_df.sort_values('_dt').tail(max_bars).reset_index(drop=True)
    if plot_df.empty:
        raise ValueError("Not enough data to render order flow chart")

    levels = int(max(4, min(26, levels)))

    min_price = float(plot_df['Low'].min())
    max_price = float(plot_df['High'].max())
    price_padding = max((max_price - min_price) * 0.025, 0.01)

    if price_precision is None:
        if max_price >= 1000:
            price_precision = 1
        elif max_price >= 100:
            price_precision = 2
        else:
            price_precision = 3

    bar_gap = 0.55
    bar_width = 1.35
    bar_positions = np.arange(len(plot_df)) * (bar_width + bar_gap)

    cells: list[dict[str, float]] = []
    annotations: list[dict[str, float]] = []
    max_volume = 0.0
    max_abs_delta = 0.0
    total_delta = 0.0

    for idx, row in plot_df.iterrows():
        high = float(row['High'])
        low = float(row['Low'])
        open_price = float(row['Open'])
        close_price = float(row['Close'])
        volume = float(row['Volume'])
        price_range = max(high - low, 1e-4)

        price_levels = np.linspace(low, high, levels + 1)
        direction = 1.0 if close_price >= open_price else -1.0
        spread = abs(close_price - open_price) / price_range
        base_volume = volume / levels if levels > 0 else volume

        # Weight more volume to middle levels, bias top/bottom by direction
        level_variation = 0.60 + 0.50 * np.cos(np.linspace(0, np.pi, levels))
        bias_gradient = np.linspace(-0.25, 0.25, levels)

        bar_buy = 0.0
        bar_sell = 0.0

        for lvl, (y0, y1, weight, bias) in enumerate(zip(price_levels[:-1], price_levels[1:], level_variation, bias_gradient)):
            lvl_volume = max(base_volume * weight, 1.0)
            buy_ratio = 0.5 + direction * 0.35 * (0.55 + spread) - direction * 0.2 * bias
            buy_ratio = float(np.clip(buy_ratio, 0.05, 0.95))
            buy_volume = lvl_volume * buy_ratio
            sell_volume = max(lvl_volume - buy_volume, 0.0)
            delta = buy_volume - sell_volume

            bar_buy += buy_volume
            bar_sell += sell_volume
            max_volume = max(max_volume, buy_volume + sell_volume)
            max_abs_delta = max(max_abs_delta, abs(delta))

            cells.append({
                "bar": idx,
                "x": bar_positions[idx],
                "y0": y0,
                "height": y1 - y0,
                "buy": buy_volume,
                "sell": sell_volume,
                "delta": delta,
                "price": (y0 + y1) / 2,
            })

        bar_delta = bar_buy - bar_sell
        total_delta += bar_delta
        annotations.append({
            "bar": idx,
            "x": bar_positions[idx],
            "high": high,
            "low": low,
            "delta": bar_delta,
            "volume": volume,
            "close": close_price,
            "open": open_price,
        })

    if max_abs_delta == 0:
        max_abs_delta = 1.0

    cmap = LinearSegmentedColormap.from_list(
        "delta_map",
        ["#E74C3C", "#2C3E50", "#27AE60"],
        N=256
    )
    norm = Normalize(vmin=-max_abs_delta, vmax=max_abs_delta)

    fig_w, fig_h = figsize
    fig = plt.figure(figsize=(max(fig_w, 16), max(fig_h, 9)))
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.86])

    subtitle = f"{symbol} · {timeframe.upper()} · Bars: {len(plot_df)}"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=15, fontweight='bold', color='#ECF0F1', y=0.99)

    for cell in cells:
        x = cell['x']
        y = cell['y0']
        width = bar_width
        height = cell['height']

        color = cmap(norm(cell['delta']))
        highlight = abs(cell['delta']) >= max_abs_delta * 0.65
        edge = '#2E4057'
        lw = 0.45
        if highlight:
            edge = '#4CAF50' if cell['delta'] > 0 else '#E74C3C'
            lw = 1.0

        rect = Rectangle((x, y), width, height, linewidth=lw, edgecolor=edge, facecolor=color, alpha=0.92)
        ax.add_patch(rect)

        sell_val = f"{int(cell['sell']):,}" if cell['sell'] >= 1 else "0"
        buy_val = f"{int(cell['buy']):,}" if cell['buy'] >= 1 else "0"

        left_x = x + width * 0.22
        right_x = x + width * 0.78
        center_x = x + width * 0.5
        mid_y = y + height * 0.5

        font_size = 7.2
        max_len = max(len(sell_val), len(buy_val))
        if max_len >= 5:
            font_size = 6.2
        if max_len >= 7:
            font_size = 5.5

        ax.text(left_x, mid_y, sell_val, ha='right', va='center', fontsize=font_size,
                color='#F26D6D', fontweight='bold', fontfamily='DejaVu Sans Mono')
        ax.text(center_x, mid_y, "×", ha='center', va='center', fontsize=font_size,
                color='#BDC3C7', fontweight='bold')
        ax.text(right_x, mid_y, buy_val, ha='left', va='center', fontsize=font_size,
                color='#8BC48A', fontweight='bold', fontfamily='DejaVu Sans Mono')

    # Annotate prices along the left-most ladder
    if cells:
        first_bar = [c for c in cells if c['bar'] == 0]
        for cell in first_bar:
            ax.text(cell['x'] - bar_gap * 1.4, cell['price'], f"{cell['price']:.{price_precision}f}",
                    ha='right', va='center', fontsize=6.4, color='#A7B1C2', alpha=0.9,
                    fontfamily='DejaVu Sans Mono')

    for ann in annotations:
        x_mid = ann['x'] + bar_width / 2
        delta_color = '#2ECC71' if ann['delta'] >= 0 else '#E74C3C'
        ax.plot([ann['x'], ann['x']], [ann['low'], ann['high']], color=delta_color, linewidth=1.1, alpha=0.85)
        ax.plot([ann['x'] + bar_width, ann['x'] + bar_width], [ann['low'], ann['high']], color=delta_color, linewidth=1.0, alpha=0.35)

        ax.text(x_mid, ann['high'] + price_padding * 0.85, f"Δ {int(ann['delta']):,}",
                ha='center', va='bottom', fontsize=8.5, color=delta_color, fontweight='bold')
        ax.text(x_mid, ann['low'] - price_padding * 0.5, f"V {int(ann['volume']):,}",
                ha='center', va='top', fontsize=7.2, color='#8596A6')

    if len(bar_positions) > 0:
        x_min = bar_positions[0] - bar_gap * 2
        x_max = bar_positions[-1] + bar_width + bar_gap * 2
    else:
        x_min, x_max = -1, 1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(min_price - price_padding * 2.5, max_price + price_padding * 3)

    time_labels = []
    for ts in plot_df['_dt']:
        if timeframe.endswith('d'):
            time_labels.append(ts.strftime('%Y-%m-%d'))
        else:
            time_labels.append(ts.strftime('%H:%M'))

    ax.set_xticks(bar_positions + bar_width / 2)
    ax.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8, color='#BDC3C7')
    ax.set_xlabel('Time', fontsize=10, color='#BDC3C7')
    ax.set_ylabel('Price', fontsize=10, color='#BDC3C7')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#34495E')
    ax.spines['bottom'].set_color('#34495E')
    ax.tick_params(axis='y', colors='#BDC3C7', labelsize=8.5)

    stats = {
        'bars': len(plot_df),
        'levels': levels,
        'max_volume': max_volume,
        'avg_delta': total_delta / len(plot_df) if len(plot_df) else 0.0,
    }

    return fig, ax, stats


def plot_overflow_matplotlib(
    data,
    x="time",
    y="value",
    threshold=None,
    title="Overflow Chart",
    xlabel="X-axis",
    ylabel="Y-axis",
    figsize=(12, 8),
    use_seaborn_theme=True,
    show_band=True,
    band_alpha=0.2,
    annotate=True,
    time_axis=None
):
    """Create a professional overflow chart using Matplotlib.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing columns `x` and `y`.
    x, y : str
        Column names for x- and y-axis.
    threshold : float | int
        Overflow threshold (horizontal reference line).
    title, ylabel, xlabel : Optional[str]
        Labels for aesthetics.
    show_band : bool
        If True, shades the region above threshold.
    band_alpha : float
        Opacity of the shaded overflow band.
    annotate : bool
        If True, annotates count and % above threshold.
    figsize : tuple[int, int]
        Figure size.
    use_seaborn_theme : bool
        If True and seaborn is installed, apply a clean theme.
    time_axis : Optional[bool]
        If True, formats x-axis as dates. If None, auto-detects by dtype.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    stats : dict
        Overflow statistics as computed by `overflow_stats`.
    """
    _validate_inputs(data, x, y, threshold)

    if use_seaborn_theme and _HAS_SNS:
        sns.set_theme(style="whitegrid")

    xvals = data[x]
    yvals = data[y]
    mask = compute_overflow_mask(yvals, threshold)
    stats = overflow_stats(yvals, threshold)

    if time_axis is None:
        time_axis = np.issubdtype(data[x].dtype, np.datetime64)

    fig, ax = plt.subplots(figsize=figsize)

    # Base line plot
    ax.plot(xvals, yvals, label=y)

    # Threshold line
    ax.axhline(y=threshold, linestyle="--", label=f"Threshold = {threshold}")

    # Overflow shading (band)
    if show_band:
        # Fill area between threshold and y where y > threshold
        ax.fill_between(xvals, threshold, yvals, where=mask, alpha=band_alpha, label="Overflow")

    # Highlight overflow points for clarity
    if mask.any():
        ax.scatter(xvals[mask], yvals[mask], s=18, label="Above threshold")

    # Labels and legend
    ax.set_title(title or "Overflow Chart")
    ax.set_ylabel(ylabel or y)
    ax.set_xlabel(xlabel or x)
    ax.legend()

    # Time axis formatting
    if time_axis:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        fig.autofmt_xdate()

    # Annotation with simple stats
    if annotate:
        note = f"over: {stats['count_over']}  ({stats['pct_over']:.1f}%)  max: {stats['max_over']:.2f}  excessΣ: {stats['total_excess']:.2f}"
        ax.text(
            0.01, 0.99, note, transform=ax.transAxes,
            ha="left", va="top"
        )

    return fig, ax, stats


def plot_overflow_plotly(
    df: pd.DataFrame,
    x: str,
    y: str,
    threshold: float | int,
    *,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    annotate: bool = True,
):
    """Create an interactive overflow chart using Plotly.

    Returns a `plotly.graph_objects.Figure`.

    Notes:
        - Requires Plotly to be installed.
    """
    if not _HAS_PLOTLY:
        raise ImportError("Plotly is not installed. Run: pip install plotly")

    _validate_inputs(df, x, y, threshold)

    fig = px.line(df, x=x, y=y, title=title or "Overflow Chart")
    fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"threshold={threshold}")

    # Shade overflow region (y > threshold)
    # We add a semi-transparent rectangle spanning the x-range and y from threshold to max
    y_max = float(pd.to_numeric(df[y], errors="coerce").max())
    if y_max > threshold:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=df[x].iloc[0],
            x1=df[x].iloc[-1],
            y0=threshold,
            y1=y_max,
            fillcolor=None,
            opacity=0.15,
            line_width=0,
        )

    # Emphasize overflow points
    mask = compute_overflow_mask(df[y], threshold)
    if mask.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, x],
                y=df.loc[mask, y],
                mode="markers",
                name="Above threshold",
            )
        )

    fig.update_layout(
        xaxis_title=(xlabel or x),
        yaxis_title=(ylabel or y),
        legend_title_text="",
    )

    if annotate:
        stats = overflow_stats(df[y], threshold)
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.99, showarrow=False,
            text=(f"over: {stats['count_over']} ({stats['pct_over']:.1f}%) · "
                  f"max: {stats['max_over']:.2f} · excessΣ: {stats['total_excess']:.2f}")
        )
    return fig


def demo_dataframe(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Generate a demo time-series DataFrame with random walk values."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01", periods=n, freq="D")
    y = np.cumsum(rng.normal(0, 1, size=n)) * 5 + 50
    return pd.DataFrame({"time": t, "value": y})


if __name__ == "__main__":
    # Minimal self-test when run directly
    df = demo_dataframe()
    fig, ax, stats = plot_overflow_matplotlib(
        df, x="time", y="value", threshold=60,
        title="Overflow Demo (Matplotlib)", ylabel="Value",
    )
    plt.show()

    if _HAS_PLOTLY:
        fig2 = plot_overflow_plotly(df, x="time", y="value", threshold=60, title="Overflow Demo (Plotly)")
        # In notebooks use fig2.show(); in scripts, write to HTML:
        fig2.write_html("overflow_demo.html")
