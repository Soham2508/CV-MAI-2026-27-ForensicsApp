"""Swap selected color channels in an image."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult

CHANNELS = ("Red", "Green", "Blue")
CHANNEL_INDICES = {name: index for index, name in enumerate(CHANNELS)}


def swap_channels(
    image: Image.Image, first_channel: str, second_channel: str
) -> Image.Image:
    """Return an RGB image with the selected color channels exchanged."""
    if first_channel not in CHANNEL_INDICES or second_channel not in CHANNEL_INDICES:
        raise ValueError("Choose two channels from Red, Green, and Blue.")
    if first_channel == second_channel:
        raise ValueError("Choose two different channels to swap.")

    order = list(range(len(CHANNELS)))
    first_index = CHANNEL_INDICES[first_channel]
    second_index = CHANNEL_INDICES[second_channel]
    order[first_index], order[second_index] = order[second_index], order[first_index]
    rgb = np.asarray(image.convert("RGB"))
    return Image.fromarray(rgb[:, :, order])


def ask_channel_pair(parent: tk.Misc) -> tuple[str, str] | None:
    """Prompt for two distinct channels, returning None when cancelled."""
    dialog = tk.Toplevel(parent)
    dialog.title("Swap color channels")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    first = tk.StringVar(value="Red")
    second = tk.StringVar(value="Blue")
    ttk.Label(dialog, text="Swap channel").grid(
        row=0, column=0, sticky="w", padx=12, pady=(12, 4)
    )
    ttk.Combobox(
        dialog, textvariable=first, values=CHANNELS, state="readonly", width=12
    ).grid(row=0, column=1, padx=(0, 12), pady=(12, 4))
    ttk.Label(dialog, text="with").grid(row=1, column=0, sticky="w", padx=12, pady=4)
    ttk.Combobox(
        dialog, textvariable=second, values=CHANNELS, state="readonly", width=12
    ).grid(row=1, column=1, padx=(0, 12), pady=4)

    result: tuple[str, str] | None = None

    def apply() -> None:
        nonlocal result
        if first.get() == second.get():
            messagebox.showerror(
                "Swap color channels",
                "Choose two different channels to swap.",
                parent=dialog,
            )
            return
        result = (first.get(), second.get())
        dialog.destroy()

    buttons = ttk.Frame(dialog)
    buttons.grid(row=2, column=0, columnspan=2, sticky="e", padx=12, pady=12)
    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Apply", command=apply).pack(side="right", padx=(0, 8))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Swap color channels"
    category = "Color & Tone"
    description = "Choose any two RGB channels to exchange."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        selected = ask_channel_pair(parent)
        if selected is None:
            return None

        first_channel, second_channel = selected
        output = swap_channels(document.current, first_channel, second_channel)
        channel_names = f"{first_channel}-{second_channel}"
        return ToolResult(
            image=output,
            message=f"Swapped {first_channel} and {second_channel} channels.",
            details={"Operation": "Channel swap", "Channels": channel_names},
        )
