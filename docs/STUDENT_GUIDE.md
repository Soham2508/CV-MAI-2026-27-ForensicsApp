# ForensicsApp student guide

This application is your starting point for the semester project. You will keep  
adding computer-vision functionality while retaining one usable desktop app.  
Each feature should be easy to demonstrate in class and easy for another person  
to find in the code.

## 1\. Setup and first run

Install Python 3.10 or newer, open a terminal in the project folder, and create  
an isolated environment:

```
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate          # Windows PowerShell
python -m pip install -r requirements.txt
python run.py
```

If a Linux installation reports `No module named tkinter`, install the Tk  
package supplied by the operating system (often `python3-tk`). Tkinter ships  
with the normal Python installers on Windows and macOS.

Open an image, try **Inspect image** and **Convert to grayscale**, then try Undo,  
Redo, Reset, and Save result. You can also start the program with:

```
python -m forensics_app
```

## 2\. Tkinter in ten minutes

Tkinter is Python's interface to the Tk desktop GUI toolkit. Four ideas explain  
most of this project.

### The root window and event loop

`tk.Tk()` creates the main window. `root.mainloop()` starts an event loop that  
waits for clicks, keystrokes, resize events, and other operating-system events.  
Do not place code after `mainloop()` and expect it to run while the window is  
open.

```python
import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Small example")
ttk.Label(root, text="Hello, vision!").pack(padx=20, pady=20)
root.mainloop()
```

The project creates the root only inside `forensics_app/app.py`. Importing a  
module therefore never opens a surprise window, and non-GUI code remains easy  
to test.

### Widgets and parents

A widget is a control such as `ttk.Label`, `ttk.Button`, `ttk.Frame`, or  
`ttk.Treeview`. Its first argument is its parent. Frames group widgets into  
sections. `ttk` widgets use the platform theme and are generally preferable to  
the older `tk` versions. A `tk.Canvas` is used where direct drawing is useful.

```python
panel = ttk.Frame(root, padding=12)
panel.pack(fill="both", expand=True)
label = ttk.Label(panel, text="Threshold")
label.pack(anchor="w")
```

### Geometry managers

Tkinter positions widgets through geometry managers:

- `pack()` is convenient for rows, columns, and filling a region;
- `grid()` is best for form-like rows and columns;
- `place()` uses coordinates and is rarely appropriate for resizable apps.

Do not mix `pack` and `grid` among children of the _same parent_. It is fine to  
use `pack` in one frame and `grid` in a nested frame. The starter uses `pack`  
for most sections and `ttk.Panedwindow` for resizable left/center/right panes.

Useful options are `fill="both"`, `expand=True`, `side="left"`, `padx`, `pady`,  
and `sticky="nsew"` (with `grid`).

### Commands, bindings, and Tk variables

A button receives a callable—not the result of calling it:

```python
def say_hello():
    print("Hello")


ttk.Button(root, text="Run", command=say_hello).pack()
```

`widget.bind("<event>", callback)` handles lower-level events. Bound callbacks  
receive an event object. `StringVar`, `IntVar`, and similar Tk variables update  
connected widgets; the status bar uses a `StringVar`.

### Dialogs

The standard modules `filedialog`, `messagebox`, and `simpledialog` cover common  
interactions. A dialog can return an empty value when the user cancels, so always  
check it.

```python
from tkinter import simpledialog

radius = simpledialog.askfloat("Blur", "Radius:", minvalue=0.0, parent=root)
if radius is None:
    return  # cancelled
```

### Images and object lifetime

Tk cannot display a PIL image directly. `ImageTk.PhotoImage` converts it. A  
reference must remain alive for as long as it is displayed; otherwise Python  
garbage-collects it and the canvas appears blank. `ImageView` stores it as  
`self._photo` for this reason. It makes a downscaled _preview_ but never resizes  
the working image.

### Keep the event loop responsive

Callbacks run on Tk's single UI thread. A long model inference blocks redraws  
and makes the app look frozen. Small filters can run directly. For expensive  
CNN, detector, or transformer work, run computation in a worker thread and use  
`root.after(...)` to update widgets back on the UI thread. Tk widgets themselves  
must not be changed from the worker thread.

## 3\. How the project is organized

```
ForensicsApp/
├── run.py                         simple launcher
├── requirements.txt               Python dependencies
├── forensics_app/
│   ├── app.py                     creates Tk and starts the application
│   ├── core/
│   │   └── image_document.py      image, original, history, load/save
│   ├── tools/
│   │   ├── base.py                common tool contract and ToolResult
│   │   ├── registry.py            ordered collection used by the sidebar
│   │   ├── image_info.py          analysis-only example
│   │   └── grayscale.py           image-producing example
│   └── ui/
│       ├── main_window.py         menus, panels, and coordination
│       └── image_view.py          scaled image preview
├── tests/                         fast tests that need no display
└── docs/
    └── STUDENT_GUIDE.md           this handout
```

The dependency direction is important:

```
UI  ─────► tools ─────► core
 │                       ▲
 └───────────────────────┘
```

`core` knows nothing about Tkinter. A tool implements one feature. The UI knows  
how to run any conforming tool, but does not know the tool's algorithm. This  
keeps image processing out of button callbacks and makes algorithms testable  
without opening a desktop window.

`ImageDocument.current` is the working PIL image. A tool returns a new image;  
the main window applies it and records history. `ImageDocument.original` remains  
unchanged so Reset is always possible.

## 4\. Add a new functionality

Every sidebar feature follows the same three-step recipe.

### Step 1: create one tool module

For example, create `forensics_app/tools/blur.py`:

```python
import tkinter as tk
from tkinter import simpledialog

from PIL import ImageFilter

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class BlurTool(ForensicsTool):
    tool_id = "gaussian_blur"  # unique, stable identifier
    title = "Gaussian blur"  # text shown in the sidebar
    category = "Filtering"
    description = "Blur the working image with a chosen radius."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        radius = simpledialog.askfloat(
            "Gaussian blur",
            "Radius (pixels):",
            parent=parent,
            minvalue=0.0,
            initialvalue=2.0,
        )
        if radius is None:
            return None

        assert document.current is not None
        output = document.current.filter(ImageFilter.GaussianBlur(radius))
        return ToolResult(
            image=output,
            message=f"Applied Gaussian blur (radius {radius:g}).",
            details={"Operation": "Gaussian blur", "Radius": radius},
        )
```

The main window already prevents image-dependent tools from running before an  
image is loaded. Returning `None` means “the user cancelled.” Returning an image  
automatically refreshes the preview and creates an undo step. `details` appears  
in the Results panel. Do not call `document.apply()` inside a tool—the window  
does that once.

### Step 2: register it

Edit `forensics_app/tools/__init__.py`:

```python
from .blur import BlurTool

# Inside build_tool_registry():
return ToolRegistry(
    [
        ImageInfoTool(),
        GrayscaleTool(),
        BlurTool(),
    ]
)
```

Restart the app. A **Filtering** section and **Gaussian blur** button will appear  
without any change to `MainWindow`.

### Step 3: test the algorithm

Separate numerical work into ordinary functions when it becomes non-trivial:

```python
# forensics_app/tools/edges.py
def compute_edges(array, low_threshold, high_threshold):
    # NumPy/OpenCV implementation
    return edge_array
```

Then let the tool class handle only dialogs, PIL/NumPy conversion, and packaging  
the result. Unit-test `compute_edges` with a small synthetic array. This makes a  
failed algorithm distinguishable from a failed GUI.

Run the starter tests at any time:

```
python -m unittest discover -s tests -v
```

## 5\. Tool patterns you will need later

An **image-producing tool** includes `image=output`; filtering, edge detection,  
segmentation overlays, bounding boxes, and saliency maps fit this pattern.

An **analysis-only tool** omits the image and fills `details`; descriptors,  
classification labels, confidence scores, and matching statistics can use it.

A **parameterized tool** asks for values with a dialog before processing, as in  
the blur example. Validate every input and make cancellation harmless.

A **model-backed tool** should load its model lazily (on first use), reuse the  
loaded model, show which weights/device are active, and do long inference away  
from the UI thread. Do not hard-code a path that only exists on your computer.

Complex results may eventually justify a dedicated window or reusable widget.  
Keep it inside that tool's module/package unless it is useful to several tools.

## 6\. Working agreements for a maintainable semester project

- Keep each feature in its own clearly named module or package.
- Never overwrite the evidence image; **Save result** uses a new filename.
- Treat the original as read-only and process a copy/current representation.
- Record parameters, model/weights, confidence, and relevant timings in Results.
- Prefer functions with inputs and return values over global variables.
- Convert color spaces explicitly. PIL commonly uses RGB; OpenCV commonly uses  
  BGR; many neural models expect normalized RGB tensors.
- Preserve aspect ratio in previews and report when an algorithm resizes input.
- Handle unsupported modes and small images with a useful error message.
- Put large datasets, checkpoints, and generated outputs outside version control.
- Add a small test for every algorithm, including at least one edge case.

## 7\. Suggested in-class demonstration checklist

For each increment, be ready to show:

1.  the app starting from a fresh checkout/environment;
2.  a representative input and the expected output;
3.  the effect of at least one parameter, where applicable;
4.  one failure or edge case handled without crashing;
5.  the feature's module and a focused algorithm test;
6.  Undo/Reset and export still working after integration.

Before presenting, run the tests and try the feature on more than the one image  
used during implementation.
