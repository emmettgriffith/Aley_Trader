#!/usr/bin/env python3
"""Label search and assignment tool with a simple Tkinter UI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import messagebox, ttk

# Generate the required label set.
LABELS: List[str] = []
for column in range(1, 5):  # c1-d1 through c4-d6
    LABELS.extend([f"c{column}-d{digit}" for digit in range(1, 7)])
LABELS.extend([f"c5-d{digit}" for digit in range(1, 5)])
LABELS.extend([f"c6-d{digit}" for digit in range(1, 5)])
LABELS.extend([f"c7-d{digit}" for digit in range(1, 5)])
LABELS.extend([f"c8-d{digit}" for digit in range(1, 4)])
LABELS.extend([f"c9-d{digit}" for digit in range(1, 4)])
LABELS.extend([f"c10-d{digit}" for digit in range(1, 4)])
LABELS.extend([f"c11-d{digit}" for digit in range(1, 3)])
LABELS.extend([f"c12-d{digit}" for digit in range(1, 3)])

ASSIGNMENTS_FILE = Path(__file__).with_name("label_assignments.json")


class AssignmentStore:
    """Loads and saves item-to-label assignments from JSON."""

    def __init__(self, storage_path: Path = ASSIGNMENTS_FILE) -> None:
        self.storage_path = storage_path
        self._items: Dict[str, Dict[str, str]] = {}
        self.load()

    def load(self) -> None:
        if not self.storage_path.exists():
            self._items = {}
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._items = {}
            return
        assignments = raw if isinstance(raw, list) else raw.get("assignments", [])
        cleaned: Dict[str, Dict[str, str]] = {}
        for entry in assignments:
            if not isinstance(entry, dict):
                continue
            item = str(entry.get("item", "")).strip()
            label = str(entry.get("label", "")).strip()
            if not item or label not in LABELS:
                continue
            cleaned[item.lower()] = {"item": item, "label": label}
        self._items = cleaned

    def save(self) -> None:
        data = [{"item": value["item"], "label": value["label"]} for value in self._items.values()]
        try:
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            messagebox.showerror("Save Error", f"Could not save assignments to {self.storage_path}")

    def get_entry(self, item: str) -> Optional[Dict[str, str]]:
        return self._items.get(item.lower())

    def set_label(self, item: str, label: str) -> None:
        clean_item = item.strip()
        if not clean_item:
            raise ValueError("Item name cannot be empty.")
        if label not in LABELS:
            raise ValueError("Invalid label.")
        self._items[clean_item.lower()] = {"item": clean_item, "label": label}
        self.save()

    def items_for_label(self, label: str) -> List[str]:
        return sorted(
            entry["item"] for entry in self._items.values() if entry["label"] == label
        )


class LabelSearchApp:
    def __init__(self) -> None:
        self.store = AssignmentStore()
        self.root = tk.Tk()
        self.root.title("Label Finder")
        self.root.geometry("480x360")
        self.root.minsize(420, 320)

        self.search_var = tk.StringVar()
        self.result_var = tk.StringVar(value="Type an item or label to get started.")
        self.new_item_var = tk.StringVar()
        self.label_select_var = tk.StringVar()

        self.current_label: Optional[str] = None
        self.current_item: Optional[str] = None

        self._build_ui()
        self._show_welcome()

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        self.root.configure(bg="#f3f3f3")

        content = ttk.Frame(self.root, padding=12)
        content.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        search_frame = ttk.Frame(content)
        search_frame.grid(row=0, column=0, sticky="ew")
        search_frame.columnconfigure(1, weight=1)

        ttk.Label(search_frame, text="Search:").grid(row=0, column=0, padx=(0, 8))
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.grid(row=0, column=1, sticky="ew")
        search_entry.bind("<Return>", self.on_search)
        ttk.Button(search_frame, text="Go", command=self.on_search).grid(row=0, column=2, padx=(8, 0))

        ttk.Separator(content).grid(row=1, column=0, pady=12, sticky="ew")

        self.result_label = ttk.Label(content, textvariable=self.result_var, wraplength=420, justify="left")
        self.result_label.grid(row=2, column=0, sticky="w")

        self.items_frame = ttk.Frame(content)
        self.items_frame.grid(row=3, column=0, pady=(10, 0), sticky="nsew")
        self.items_frame.columnconfigure(0, weight=1)
        content.rowconfigure(3, weight=1)

        ttk.Label(self.items_frame, text="Items with this label:").grid(row=0, column=0, sticky="w")
        self.items_list = tk.Listbox(self.items_frame, height=6, activestyle="none")
        self.items_list.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        self.items_frame.rowconfigure(1, weight=1)

        self.label_assign_frame = ttk.Frame(content, padding=(0, 8, 0, 0))
        self.label_assign_frame.grid(row=4, column=0, sticky="ew")
        self.label_assign_frame.columnconfigure(0, weight=1)
        self.current_label_var = tk.StringVar()
        ttk.Label(
            self.label_assign_frame,
            textvariable=self.current_label_var,
            font=("Helvetica", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")

        assign_entry = ttk.Entry(self.label_assign_frame, textvariable=self.new_item_var)
        assign_entry.grid(row=1, column=0, sticky="ew", pady=(6, 4))
        assign_entry.bind("<Return>", self.assign_to_current_label)
        ttk.Button(
            self.label_assign_frame, text="Assign Item", command=self.assign_to_current_label
        ).grid(row=1, column=1, padx=(8, 0))

        self.item_assign_frame = ttk.Frame(content, padding=(0, 8, 0, 0))
        self.item_assign_frame.grid(row=5, column=0, sticky="ew")
        self.item_assign_frame.columnconfigure(0, weight=1)

        self.item_assign_label_var = tk.StringVar()
        ttk.Label(
            self.item_assign_frame,
            textvariable=self.item_assign_label_var,
            font=("Helvetica", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")

        self.label_selector = ttk.Combobox(
            self.item_assign_frame,
            textvariable=self.label_select_var,
            values=LABELS,
            state="readonly",
        )
        self.label_selector.grid(row=1, column=0, sticky="ew", pady=(6, 4))
        self.label_selector.bind("<Return>", self.assign_label_to_current_item)
        ttk.Button(
            self.item_assign_frame, text="Set Label", command=self.assign_label_to_current_item
        ).grid(row=1, column=1, padx=(8, 0))

    def _show_welcome(self) -> None:
        self.current_label = None
        self.current_item = None
        self.items_frame.grid_remove()
        self.label_assign_frame.grid_remove()
        self.item_assign_frame.grid_remove()

    def on_search(self, event: Optional[tk.Event] = None) -> None:
        query = self.search_var.get().strip()
        if not query:
            self.result_var.set("Type an item or label to get started.")
            self._show_welcome()
            return

        label = self._match_label(query)
        if label:
            self._show_label_view(label)
            return

        entry = self.store.get_entry(query)
        if entry:
            self._show_item_view(entry["item"], entry["label"])
            return

        self._show_unassigned_item(query)

    def _match_label(self, query: str) -> Optional[str]:
        lowered = query.lower()
        exact_matches = [label for label in LABELS if label.lower() == lowered]
        if exact_matches:
            return exact_matches[0]
        prefix_matches = [label for label in LABELS if label.lower().startswith(lowered)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        return None

    def _show_label_view(self, label: str) -> None:
        self.current_label = label
        self.current_item = None
        items = self.store.items_for_label(label)
        self.result_var.set(
            f"Label '{label}' currently has {len(items)} item(s)."
        )
        self._refresh_list(items)
        self.items_frame.grid()
        self.label_assign_frame.grid()
        self.item_assign_frame.grid_remove()
        self.current_label_var.set(f"Assign a new item to '{label}'")
        self.new_item_var.set("")

    def _refresh_list(self, items: List[str]) -> None:
        self.items_list.delete(0, tk.END)
        for entry in items:
            self.items_list.insert(tk.END, entry)

    def _show_item_view(self, item: str, label: str) -> None:
        self.current_item = item
        self.current_label = None
        self.result_var.set(f"Item '{item}' is assigned to label '{label}'.")
        self.items_frame.grid_remove()
        self.label_assign_frame.grid_remove()
        self.item_assign_frame.grid()
        self.item_assign_label_var.set(f"Change label for '{item}'")
        self.label_select_var.set(label)

    def _show_unassigned_item(self, item: str) -> None:
        self.current_item = item
        self.current_label = None
        self.result_var.set(f"No label is assigned to '{item}'. Choose one below.")
        self.items_frame.grid_remove()
        self.label_assign_frame.grid_remove()
        self.item_assign_frame.grid()
        self.item_assign_label_var.set(f"Assign a label to '{item}'")
        self.label_select_var.set("")

    def assign_to_current_label(self, event: Optional[tk.Event] = None) -> None:
        if self.current_label is None:
            return
        item = self.new_item_var.get().strip()
        if not item:
            messagebox.showinfo("Assign Item", "Type an item name before assigning.")
            return
        try:
            self.store.set_label(item, self.current_label)
        except ValueError as exc:
            messagebox.showerror("Assign Item", str(exc))
            return
        self.new_item_var.set("")
        self._show_label_view(self.current_label)

    def assign_label_to_current_item(self, event: Optional[tk.Event] = None) -> None:
        if self.current_item is None:
            return
        label = self.label_select_var.get()
        if label not in LABELS:
            messagebox.showinfo("Set Label", "Please select a valid label.")
            return
        try:
            self.store.set_label(self.current_item, label)
        except ValueError as exc:
            messagebox.showerror("Set Label", str(exc))
            return
        self._show_item_view(self.current_item, label)


def main() -> None:
    app = LabelSearchApp()
    app.run()


if __name__ == "__main__":
    main()
