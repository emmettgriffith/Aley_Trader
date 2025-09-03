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
