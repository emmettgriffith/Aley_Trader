import tkinter as tk
from tkinter import ttk

TOOL_LIST = [
    ("trend", "Trendline", "📈"),
    ("hline", "Horizontal Line", "━"),
    ("vline", "Vertical Line", "┃"),
    ("fibo", "Fibonacci", "𝑭"),
    ("text", "Text", "📝"),
    ("rect", "Rectangle", "▭"),
    ("ellipse", "Ellipse", "◯"),
    ("triangle", "Triangle", "△"),
    ("freehand", "Freehand", "✏️"),
    ("erase", "Eraser", "🧹"),
    ("zoom", "Zoom", "🔍"),
    ("crosshair", "Crosshair", "+")
]

class DrawingToolbar(ttk.Frame):
    def __init__(self, parent, on_select_tool):
        super().__init__(parent, width=56)
        self.pack_propagate(False)
        self.on_select_tool = on_select_tool
        self.selected_tool = None
        self.tool_buttons = {}
        for tool_key, tool_name, icon in TOOL_LIST:
            btn = ttk.Button(self, text=icon, width=3, command=lambda t=tool_key: self.select_tool(t))
            btn.pack(side=tk.TOP, pady=3, padx=2)
            btn.tooltip = tool_name  # For future tooltip support
            self.tool_buttons[tool_key] = btn
    def select_tool(self, tool_key):
        self.selected_tool = tool_key
        self.on_select_tool(tool_key)
        for t, btn in self.tool_buttons.items():
            btn.state(["!selected"])
        self.tool_buttons[tool_key].state(["selected"])
