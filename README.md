# ForensicsApp course starter

ForensicsApp is the semester project shell for the MSc Computer Vision course.  
Students add a small, demonstrable image-forensics feature as the course moves  
from classical image processing to deep learning and interpretability.

The starter already provides:

- a Tkinter desktop interface with tool, image, and results areas;
- opening and exporting common image formats;
- a fit-to-window image preview;
- undo, redo, and reset-to-original behavior;
- a tiny plug-in-style contract for adding course functionality;
- one analysis example and one image-processing example.

## Quick start

Python 3.10 or newer is recommended.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

On some Linux distributions, Tkinter is a separate system package. For example,  
Ubuntu/Debian users may first need `sudo apt install python3-tk`.

The complete student handout is [docs/STUDENT_GUIDE.md](docs/STUDENT_GUIDE.md).

## Useful commands

```
python -m forensics_app
python -m unittest discover -s tests -v
python -m compileall forensics_app
```

## Object detection tools (Set 9)

Set 9 adds object detection with Ultralytics YOLO. `ultralytics` (with PyTorch)
is part of `requirements.txt`. Detection runs on the Apple GPU (`mps`), CUDA or
the CPU; the tools pick the device automatically.

| Step | Where | What it does |
|---|---|---|
| 9.1 Pretrained detector | Sidebar → Deep Learning → **YOLO Object Detection** | Runs YOLO11n pretrained on COCO (80 classes) on the open image; thresholds, input size and a class filter are adjustable. The weights (`models/yolo/pretrained/yolo11n.pt`, git-ignored) are downloaded on first use |
| 9.2 Train | `notebooks/train_yolo.ipynb` | Converts the Roboflow blood cell dataset from COCO to YOLO format, explores it, fine-tunes YOLO11n (plus a from-scratch baseline), evaluates on the test split (Ultralytics metrics and PR / FPPI curves and mAP implemented from scratch) and exports both models |
| 9.3 Blood cells | Sidebar → Deep Learning → **Blood Cell Detection** | Detects platelets (PC), red (RBC) and white (WBC) blood cells with an exported model and draws ground truth (dashed) and predictions (solid) on the same image, with false positives and missed cells marked and counted; compares both models side by side; evaluates a whole folder of labelled images (per-class precision, recall and F1, the hardest images, a CSV in `outputs/set9_3_detection/`); can leave unmatched cells cut by the image border out of the counts |

The dataset is not committed. Download the COCO export of
[blood-cell-st9na](https://universe.roboflow.com/tumor-4iauk/blood-cell-st9na)
(CC BY 4.0) and unzip it into `datasets/blood_cell/coco/`; the notebook writes
the YOLO version to `datasets/blood_cell/yolo/`. The Roboflow YOLO export, if
unzipped into `datasets/blood_cell/roboflow_yolo/`, is used to check the
conversion.

Each exported model folder in `models/yolo/` contains:

| File | Contents |
|---|---|
| `best.pt` | Trained weights (best validation epoch) |
| `passport.json` | Dataset, architecture, training settings, test metrics, false-positive analysis and inference defaults |
| `training_curves.png` | Ultralytics training curves (`results.png`) |
| `results.csv`, `train_args.yaml` | Per-epoch metrics and the full training configuration |

`models/yolo/blood_cell_samples/` holds 12 test images with their YOLO labels,
so the Blood Cell Detection tool can show ground truth without the dataset. For
any other image it looks for `labels/<name>.txt` next to an `images/` folder, or
`<name>.txt` next to the image.

## Course roadmap

The shell is intentionally neutral about the algorithms. Possible tool modules  
can follow the syllabus: filtering and edges, HOG, SIFT, bag of visual words,  
face detection/recognition, CNN classification and detection, transformers,  
semantic segmentation, and visualization/interpretability. Model weights and  
datasets should not be committed to the starter repository; document where they  
come from and keep paths configurable.

