"""Reusable dialog for parameters with slider and numeric input controls."""

from __future__ import annotations

from collections.abc import Callable
import math
import tkinter as tk
from tkinter import messagebox, ttk

ParameterField = tuple[str, str, float, float, float, float]


def ask_parameters(
    parent: tk.Misc,
    title: str,
    fields: tuple[ParameterField, ...],
    validator: Callable[[dict[str, float]], str | None] | None = None,
) -> dict[str, float] | None:
    """Show slider and input controls, returning values or None if cancelled."""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()
    controls: dict[str, ttk.Spinbox] = {}
    bounds: dict[str, tuple[str, float, float]] = {}

    for row, field in enumerate(fields):
        key, label, minimum, maximum, step, initial = field
        bounds[key] = (label, minimum, maximum)
        variable = tk.DoubleVar(value=initial)
        ttk.Label(dialog, text=label).grid(
            row=row * 2, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 0)
        )
        tk.Scale(
            dialog,
            from_=minimum,
            to=maximum,
            resolution=step,
            orient="horizontal",
            variable=variable,
            showvalue=False,
            length=260,
        ).grid(row=row * 2 + 1, column=0, sticky="ew", padx=(12, 4))
        control = ttk.Spinbox(
            dialog,
            from_=minimum,
            to=maximum,
            increment=step,
            textvariable=variable,
            width=8,
        )
        control.grid(row=row * 2 + 1, column=1, padx=(4, 12))
        controls[key] = control

    result: dict[str, float] | None = None

    def apply() -> None:
        nonlocal result
        try:
            values = {key: float(control.get()) for key, control in controls.items()}
        except ValueError:
            messagebox.showerror(title, "Enter a valid number.", parent=dialog)
            return
        for key, (label, minimum, maximum) in bounds.items():
            value = values[key]
            if not math.isfinite(value) or not minimum <= value <= maximum:
                messagebox.showerror(
                    title,
                    f"{label} must be between {minimum:g} and {maximum:g}.",
                    parent=dialog,
                )
                return
        error = validator(values) if validator else None
        if error:
            messagebox.showerror(title, error, parent=dialog)
            return
        result = values
        dialog.destroy()

    buttons = ttk.Frame(dialog)
    buttons.grid(
        row=len(fields) * 2,
        column=0,
        columnspan=2,
        sticky="e",
        padx=12,
        pady=12,
    )
    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Apply", command=apply).pack(side="right", padx=(0, 8))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result
