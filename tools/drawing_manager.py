import matplotlib.pyplot as plt

class DrawingManager:
    def __init__(self, chart_area):
        self.chart_area = chart_area
        self.current_tool = None
        self.start_coords = None
        self.temp_artist = None
        self.drawings = []
    def set_tool(self, tool):
        self.current_tool = tool
    def on_press(self, event):
        if self.current_tool == "trend":
            self.start_coords = (event.xdata, event.ydata)
            self.temp_artist, = self.chart_area.ax.plot([event.xdata], [event.ydata], color='blue')
        elif self.current_tool == "rect":
            self.start_coords = (event.xdata, event.ydata)
            self.temp_artist = self.chart_area.ax.add_patch(
                self.chart_area.ax.figure.canvas.figure.gca().add_patch(
                    plt.Rectangle((event.xdata, event.ydata), 0, 0, fill=False, color='red')))
        # TODO: Add logic for other tools
    def on_move(self, event):
        if self.temp_artist and self.start_coords and event.xdata and event.ydata:
            if self.current_tool == "trend":
                x0, y0 = self.start_coords
                self.temp_artist.set_data([x0, event.xdata], [y0, event.ydata])
                self.chart_area.canvas.draw_idle()
            elif self.current_tool == "rect":
                x0, y0 = self.start_coords
                width = event.xdata - x0
                height = event.ydata - y0
                self.temp_artist.set_width(width)
                self.temp_artist.set_height(height)
                self.chart_area.canvas.draw_idle()
    def on_release(self, event):
        if self.temp_artist:
            self.drawings.append(self.temp_artist)
            self.temp_artist = None
            self.start_coords = None
            self.chart_area.canvas.draw_idle()
        # TODO: Add logic for persistence, edit, erase, etc.
