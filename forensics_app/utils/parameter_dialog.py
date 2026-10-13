"""Reusable dialog for parameters with slider and numeric input controls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import tkinter as tk
from tkinter import messagebox, ttk


@dataclass(frozen=True)
class ParameterField:
    key: str
    label: str
    minimum: float
    maximum: float
    step: float
    initial: float
    show_slider: bool = True
    show_input: bool = True
    no_threshold: bool = False


LegacyParameterField = tuple[str, str, float, float, float, float]
ParameterSpec = ParameterField | LegacyParameterField


def ask_parameters(
    parent: tk.Misc,
    title: str,
    fields: tuple[ParameterSpec, ...],
    validator: Callable[[dict[str, float]], str | None] | None = None,
) -> dict[str, float] | None:
    """Show slider and input controls, returning values or None if cancelled."""
    normalized_fields: list[ParameterField] = []
    for field in fields:
        if isinstance(field, ParameterField):
            parameter = field
        else:
            parameter = ParameterField(*field)
        if not parameter.show_slider and not parameter.show_input:
            raise ValueError(f"Parameter {parameter.key!r} must have a control.")
        normalized_fields.append(parameter)

    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()
    controls: dict[str, ttk.Spinbox] = {}
    variables: dict[str, tk.DoubleVar] = {}

    for row, field in enumerate(normalized_fields):
        key = field.key
        label = field.label
        minimum = field.minimum
        maximum = field.maximum
        step = field.step
        variable = tk.DoubleVar(value=field.initial)
        variables[key] = variable
        ttk.Label(dialog, text=label).grid(
            row=row * 2, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 0)
        )
        if field.show_slider:
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
        if field.show_input:
            input_options: dict[str, float] = {"increment": step}
            if not field.no_threshold:
                input_options.update(from_=minimum, to=maximum)
            control = ttk.Spinbox(
                dialog,
                textvariable=variable,
                width=8,
                **input_options,
            )
            control.grid(
                row=row * 2 + 1,
                column=1 if field.show_slider else 0,
                columnspan=1 if field.show_slider else 2,
                padx=(4, 12) if field.show_slider else (12, 12),
            )
            controls[key] = control

    result: dict[str, float] | None = None

    def apply() -> None:
        nonlocal result
        try:
            values = {
                field.key: float(
                    controls[field.key].get()
                    if field.show_input
                    else variables[field.key].get()
                )
                for field in normalized_fields
            }
        except (ValueError, tk.TclError):
            messagebox.showerror(title, "Enter a valid number.", parent=dialog)
            return
        for field in normalized_fields:
            value = values[field.key]
            if (
                not math.isfinite(value)
                or value < field.minimum
                or (not field.no_threshold and value > field.maximum)
            ):
                bounds_message = (
                    f"{field.label} must be at least {field.minimum:g}."
                    if field.no_threshold
                    else (
                        f"{field.label} must be between "
                        f"{field.minimum:g} and {field.maximum:g}."
                    )
                )
                messagebox.showerror(
                    title,
                    bounds_message,
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
        row=len(normalized_fields) * 2,
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
