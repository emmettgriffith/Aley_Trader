import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class ChartArea(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._press_cb = None
        self._move_cb = None
        self._release_cb = None
    def connect_matplotlib_events(self, on_press, on_move, on_release):
        self._press_cb = self.figure.canvas.mpl_connect('button_press_event', on_press)
        self._move_cb = self.figure.canvas.mpl_connect('motion_notify_event', on_move)
        self._release_cb = self.figure.canvas.mpl_connect('button_release_event', on_release)
    def disconnect_events(self):
        for cb in [self._press_cb, self._move_cb, self._release_cb]:
            if cb:
                self.figure.canvas.mpl_disconnect(cb)
