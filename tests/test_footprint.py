import unittest

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency for tests
    pd = None

try:
    from footprint_chart import build_footprints, render_footprint
except Exception:  # pragma: no cover - skip tests if dependencies missing
    build_footprints = None
    render_footprint = None


def make_ticks(prices, sizes, sides, start="2025-01-01 09:30:00Z"):
    ts = pd.date_range(start=start, periods=len(prices), freq="5S", tz="UTC")
    return pd.DataFrame({
        "ts": ts,
        "price": prices,
        "size": sizes,
        "side": sides,
    })


@unittest.skipIf(pd is None or build_footprints is None, "footprint dependencies not available")
class FootprintAggregationTests(unittest.TestCase):
    def test_cell_binning_creates_contiguous_levels(self):
        prices = [100.00, 100.01, 100.02, 100.02]
        sizes = [10, 20, 30, 40]
        sides = ["ASK", "BID", "ASK", "BID"]
        ticks = make_ticks(prices, sizes, sides)

        bars = build_footprints(ticks, interval="1min", tick_size=0.01)
        self.assertEqual(len(bars), 1)
        cell_prices = [cell.price for cell in bars[0].cells]
        self.assertEqual(cell_prices, sorted(cell_prices, reverse=True))
        self.assertAlmostEqual(cell_prices[0] - cell_prices[-1], 0.02)

    def test_bid_ask_sums_per_bar(self):
        prices = [50.0, 50.0, 50.25, 50.25]
        sizes = [5, 10, 15, 20]
        sides = ["BID", "ASK", "BID", "ASK"]
        ticks = make_ticks(prices, sizes, sides)

        bars = build_footprints(ticks, interval="1min", tick_size=0.25)
        self.assertEqual(len(bars), 1)
        totals = {cell.price: (cell.bid_vol, cell.ask_vol) for cell in bars[0].cells}
        self.assertEqual(totals[50.25], (15, 20))
        self.assertEqual(totals[50.0], (5, 10))

    def test_render_handles_missing_levels(self):
        prices = [80.0, 81.0]
        sizes = [10, 12]
        sides = ["ASK", "BID"]
        ticks = make_ticks(prices, sizes, sides)

        bars = build_footprints(ticks, interval="1min", tick_size=0.5)
        self.assertTrue(len(bars[0].cells) >= 3)
        mid_cell = next(cell for cell in bars[0].cells if abs(cell.price - 80.5) < 1e-9)
        self.assertEqual(mid_cell.bid_vol, 0)
        self.assertEqual(mid_cell.ask_vol, 0)

        fig = render_footprint(ticks, tick_size=0.5, interval="1min")
        fig.canvas.draw()
        import matplotlib.pyplot as plt

        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
