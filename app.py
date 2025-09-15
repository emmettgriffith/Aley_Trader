import tkinter as tk
from tkinter import ttk
from ui.toolbar import DrawingToolbar
from ui.chart_area import ChartArea
from tools.drawing_manager import DrawingManager

def main():
    root = tk.Tk()
    root.title("Aley Trader - Chart Drawing Demo")
    root.geometry("1200x800")
    container = ttk.Frame(root)
    container.pack(fill=tk.BOTH, expand=True)
    # Toolbar (left)
    def on_select_tool(tool_key):
        drawing_manager.set_tool(tool_key)
    toolbar = DrawingToolbar(container, on_select_tool)
    toolbar.pack(side=tk.LEFT, fill=tk.Y, expand=False)
    # Chart area (center)
    chart_area = ChartArea(container)
    chart_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    # Drawing manager
    drawing_manager = DrawingManager(chart_area)
    chart_area.connect_matplotlib_events(drawing_manager.on_press, drawing_manager.on_move, drawing_manager.on_release)
    root.mainloop()

if __name__ == "__main__":
    main()
