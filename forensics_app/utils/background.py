"""Run slow jobs in a worker thread while the window stays responsive.

Training a cascade, building eigenfaces (including the LFW download) and
searching a folder used to run on the Tkinter thread, freezing the whole app
with no feedback. ``run_with_progress`` runs such a job in a worker thread,
shows a small progress window with the elapsed time, and returns the job's
result (or re-raises its exception) on the main thread.

The job must not touch Tkinter objects: read settings from widgets before
calling, and update widgets after it returns. Figures must be rendered with
``matplotlib.figure.Figure`` rather than pyplot.
"""

from __future__ import annotations

from collections.abc import Callable
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Any, TypeVar

T = TypeVar("T")

POLL_MS = 100


def run_with_progress(
    parent: Any,
    title: str,
    message: str,
    job: Callable[..., T],
    /,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run ``job(*args, **kwargs)`` in a worker thread behind a progress window.

    Blocks the calling Tkinter code (like a modal dialog) while the event loop
    keeps running, so the app keeps redrawing. Without a real Tk parent (for
    example in unit tests), the job runs directly in the calling thread.
    """
    if not isinstance(parent, tk.Misc):
        return job(*args, **kwargs)

    outcome: dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["value"] = job(*args, **kwargs)
        except BaseException as exc:  # re-raised on the main thread below
            outcome["error"] = exc

    previous_grab = parent.grab_current()
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent.winfo_toplevel())
    dialog.resizable(False, False)
    ttk.Label(dialog, text=message, wraplength=320).pack(padx=18, pady=(16, 8))
    bar = ttk.Progressbar(dialog, mode="indeterminate", length=300)
    bar.pack(padx=18, pady=4)
    elapsed = tk.StringVar(value="Elapsed: 0 s")
    ttk.Label(dialog, textvariable=elapsed).pack(padx=18, pady=(4, 6))
    ttk.Label(
        dialog,
        text="Please wait; this window closes when the job finishes.",
        foreground="#666666",
    ).pack(padx=18, pady=(0, 14))
    # The job cannot be interrupted safely, so the window cannot be closed.
    dialog.protocol("WM_DELETE_WINDOW", lambda: None)
    bar.start(12)
    dialog.grab_set()

    thread = threading.Thread(target=worker, daemon=True)
    started = time.monotonic()
    thread.start()

    def poll() -> None:
        if thread.is_alive():
            elapsed.set(f"Elapsed: {time.monotonic() - started:.0f} s")
            dialog.after(POLL_MS, poll)
            return
        bar.stop()
        dialog.grab_release()
        dialog.destroy()

    dialog.after(POLL_MS, poll)
    parent.wait_window(dialog)

    # Give the modal grab back to the dialog that had it (e.g. a trainer window).
    if previous_grab is not None:
        try:
            if previous_grab.winfo_exists():
                previous_grab.grab_set()
        except tk.TclError:
            pass

    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
