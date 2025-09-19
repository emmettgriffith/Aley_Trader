#!/usr/bin/env python3
"""Simple windowed notes application with an Apple Notes-inspired layout."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import messagebox, ttk

NOTES_FILE = Path(__file__).with_name("notes_data.json")
AUTOSAVE_DELAY_MS = 600


@dataclass
class Note:
    title: str
    body: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Note":
        return Note(
            title=str(data.get("title", "")),
            body=str(data.get("body", "")),
            created_at=str(data.get("created_at", datetime.now().isoformat(timespec="seconds"))),
        )


def load_notes() -> List[Note]:
    if not NOTES_FILE.exists():
        return []
    try:
        raw_data = json.loads(NOTES_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw_data, list):
            return []
        return [Note.from_dict(item) for item in raw_data]
    except (json.JSONDecodeError, OSError):
        return []


def save_notes(notes: List[Note]) -> None:
    serialized = [note.to_dict() for note in notes]
    NOTES_FILE.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


class NotesApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Notes")
        self.root.geometry("840x520")
        self.root.minsize(640, 420)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.notes: List[Note] = load_notes()
        self.current_index: Optional[int] = None
        self._autosave_job: Optional[str] = None
        self._suppress_list_events = False

        self._build_ui()
        self._populate_notes()

        if self.notes:
            self._select_note(0)
        else:
            self._clear_editor()
            self._set_editor_state(False)

    def _build_ui(self) -> None:
        self.root.configure(bg="#f5f5f5")
        style = ttk.Style(self.root)
        style.configure("Sidebar.TFrame", background="#efefef")
        style.configure("SidebarTitle.TLabel", font=("Helvetica", 13, "bold"), background="#efefef")
        style.configure("Editor.TFrame", background="#ffffff")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=0, minsize=230)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        # Sidebar with note list and actions
        sidebar = ttk.Frame(container, style="Sidebar.TFrame", padding=(8, 10, 8, 10))
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1)

        header = ttk.Frame(sidebar, style="Sidebar.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Notes", style="SidebarTitle.TLabel").grid(row=0, column=0, sticky="w")
        actions = ttk.Frame(header, style="Sidebar.TFrame")
        actions.grid(row=0, column=1, sticky="e")

        new_button = ttk.Button(actions, text="New", command=self.create_note)
        new_button.grid(row=0, column=0, padx=(0, 6))
        delete_button = ttk.Button(actions, text="Delete", command=self.delete_note)
        delete_button.grid(row=0, column=1)

        list_frame = ttk.Frame(sidebar, style="Sidebar.TFrame")
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.notes_list = tk.Listbox(
            list_frame,
            activestyle="none",
            exportselection=False,
            selectmode=tk.SINGLE,
            highlightthickness=0,
            relief=tk.FLAT,
            font=("Helvetica", 11),
            background="#f7f7f7",
        )
        self.notes_list.grid(row=0, column=0, sticky="nsew")
        self.notes_list.bind("<<ListboxSelect>>", self._on_list_select)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.notes_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.notes_list.configure(yscrollcommand=scrollbar.set)

        # Editor area
        editor = ttk.Frame(container, style="Editor.TFrame", padding=(12, 10, 12, 12))
        editor.grid(row=0, column=1, sticky="nsew")
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(2, weight=1)

        self.title_var = tk.StringVar()
        self.title_entry = ttk.Entry(editor, textvariable=self.title_var, font=("Helvetica", 16, "bold"))
        self.title_entry.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.title_entry.bind("<KeyRelease>", self._on_editor_change)

        ttk.Separator(editor).grid(row=1, column=0, sticky="ew", pady=(0, 8))

        body_frame = ttk.Frame(editor, padding=0)
        body_frame.grid(row=2, column=0, sticky="nsew")
        body_frame.columnconfigure(0, weight=1)
        body_frame.rowconfigure(0, weight=1)

        self.body_text = tk.Text(
            body_frame,
            wrap="word",
            font=("Helvetica", 12),
            undo=True,
            relief=tk.FLAT,
            borderwidth=0,
            background="#ffffff",
        )
        self.body_text.grid(row=0, column=0, sticky="nsew")
        self.body_text.bind("<KeyRelease>", self._on_editor_change)
        self.body_text.bind("<<Modified>>", self._on_text_modified)

        body_scroll = ttk.Scrollbar(body_frame, orient="vertical", command=self.body_text.yview)
        body_scroll.grid(row=0, column=1, sticky="ns")
        self.body_text.configure(yscrollcommand=body_scroll.set)

    def _populate_notes(self) -> None:
        self._suppress_list_events = True
        self.notes_list.delete(0, tk.END)
        for note in self.notes:
            self.notes_list.insert(tk.END, self._format_list_label(note.title, note.body))
        self._suppress_list_events = False

    def _select_note(self, index: Optional[int]) -> None:
        if index is not None and not (0 <= index < len(self.notes)):
            index = None

        if self.current_index is not None and self.current_index != index:
            self._flush_pending_save()
            self._save_current_note()

        self._suppress_list_events = True
        self.notes_list.selection_clear(0, tk.END)
        if index is not None:
            self.notes_list.selection_set(index)
            self.notes_list.activate(index)
            self.notes_list.see(index)
        self._suppress_list_events = False

        if index is None:
            self.current_index = None
            self._clear_editor()
            self._set_editor_state(False)
            return

        self.current_index = index
        self._set_editor_state(True)
        self._load_note_into_editor(self.notes[index])

    def _load_note_into_editor(self, note: Note) -> None:
        self.title_entry.configure(state="normal")
        self.title_var.set(note.title)
        self.body_text.configure(state="normal")
        self.body_text.delete("1.0", tk.END)
        self.body_text.insert("1.0", note.body)
        self.body_text.edit_modified(False)
        self.body_text.mark_set("insert", "1.0")

    def _clear_editor(self) -> None:
        self.title_entry.configure(state="normal")
        self.title_var.set("")
        self.body_text.configure(state="normal")
        self.body_text.delete("1.0", tk.END)
        self.body_text.edit_modified(False)

    def _set_editor_state(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.title_entry.configure(state=state)
        self.body_text.configure(state=state)

    def _on_list_select(self, event: tk.Event) -> None:  # type: ignore[override]
        if self._suppress_list_events:
            return
        selection = event.widget.curselection()
        if not selection:
            return
        index = selection[0]
        if index == self.current_index:
            return
        self._flush_pending_save()
        self._save_current_note()
        self.current_index = index
        self._set_editor_state(True)
        self._load_note_into_editor(self.notes[index])

    def _on_editor_change(self, _event: tk.Event) -> None:  # type: ignore[override]
        if self.current_index is None:
            return
        self._schedule_autosave()
        title, body = self._peek_editor_content()
        self._update_list_preview(self.current_index, title, body)

    def _on_text_modified(self, event: tk.Event) -> None:  # type: ignore[override]
        if self.body_text.edit_modified():
            self.body_text.edit_modified(False)
            self._on_editor_change(event)

    def _schedule_autosave(self) -> None:
        self._flush_pending_save()
        self._autosave_job = self.root.after(AUTOSAVE_DELAY_MS, self._autosave_now)

    def _flush_pending_save(self) -> None:
        if self._autosave_job is not None:
            self.root.after_cancel(self._autosave_job)
            self._autosave_job = None

    def _autosave_now(self) -> None:
        self._autosave_job = None
        self._save_current_note()

    def _save_current_note(self) -> None:
        if self.current_index is None:
            return
        title, body = self._peek_editor_content()
        note = self.notes[self.current_index]
        if note.title == title and note.body == body:
            return
        note.title = title
        note.body = body
        save_notes(self.notes)
        self._update_list_preview(self.current_index, title, body)

    def _peek_editor_content(self) -> tuple[str, str]:
        title = self.title_var.get().strip()
        body = self.body_text.get("1.0", "end-1c").rstrip()
        return title, body

    def _update_list_preview(self, index: int, title: str, body: str) -> None:
        if not (0 <= index < self.notes_list.size()):
            return
        label = self._format_list_label(title, body)
        current_selection = self.notes_list.curselection()
        self.notes_list.delete(index)
        self.notes_list.insert(index, label)
        if current_selection:
            for sel in current_selection:
                self.notes_list.selection_set(sel)

    @staticmethod
    def _format_list_label(title: str, body: str) -> str:
        title_display = title.strip()
        body_display = body.strip().splitlines()[0] if body.strip() else ""
        if not title_display and not body_display:
            return "(untitled)"
        if not title_display:
            title_display = body_display or "(untitled)"
        label = title_display
        if body_display and body_display != title_display:
            snippet = body_display[:50]
            if len(body_display) > 50:
                snippet += "…"
            label = f"{title_display} — {snippet}"
        return label

    def create_note(self) -> None:
        self._flush_pending_save()
        self._save_current_note()
        new_note = Note(title="", body="")
        self.notes.insert(0, new_note)
        self._populate_notes()
        self._select_note(0)
        self.title_entry.focus_set()
        save_notes(self.notes)

    def delete_note(self) -> None:
        if self.current_index is None:
            return
        note = self.notes[self.current_index]
        title_preview = note.title or note.body.splitlines()[0] if note.body else "untitled note"
        if not messagebox.askyesno("Delete Note", f"Delete '{title_preview or 'untitled note'}'?", parent=self.root):
            return
        self._flush_pending_save()
        del self.notes[self.current_index]
        save_notes(self.notes)
        if not self.notes:
            self._populate_notes()
            self._select_note(None)
            return
        next_index = min(self.current_index, len(self.notes) - 1)
        self._populate_notes()
        self._select_note(next_index)

    def on_close(self) -> None:
        self._flush_pending_save()
        self._save_current_note()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = NotesApp()
    app.run()


if __name__ == "__main__":
    main()
