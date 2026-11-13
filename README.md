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

## Deep learning tools (Set 8)

Set 8 adds CNN image classification on CIFAR-10. Its dependencies
(`tensorflow`, `umap-learn`, and `ipykernel` for the notebook) are part of
`requirements.txt`, so the Quick start installs them.

| Step | Where | What it does |
|---|---|---|
| 8.1 Train | `notebooks/train_cnn.ipynb` | Trains a small CNN from scratch and an InceptionV3 transfer-learning model, and exports each one to `models/cnn/<model_name>/` |
| 8.2 Classify | Sidebar → Deep Learning → **CNN Classification** | Lists the exported models, shows their model passport, compares them, and classifies an image (samples in `models/cnn/sample_images/`) |
| 8.3 Project features | Sidebar → Deep Learning → **CNN Feature Projection** | Loads the saved test-set feature vectors and projects them to 2-D with t-SNE and UMAP, with cluster-quality scores and a two-model comparison |

Each exported model folder contains:

| File | Contents |
|---|---|
| `model.keras` | Trained weights (Keras 3 format) |
| `passport.json` | Dataset, sample sizes, class names, preprocessing, training history and test evaluation |
| `features_test.npz` | Feature vectors of the 10,000 CIFAR-10 test images (`features` layer), with true and predicted labels |
| `training_curves.png` | Loss and accuracy per epoch |

Everything is committed except the InceptionV3 `model.keras` (162 MB, over
GitHub's file limit). To classify with InceptionV3, unzip the shared weights so
that `models/cnn/cifar10_inceptionv3/model.keras` exists, or rerun Step 6 of the
notebook. Without it the classification tool reports the missing weights; the
passport and the feature projection still work for both models.

The feature projection does not run the CNN: it only needs `features_test.npz`,
scikit-learn and, for UMAP, `umap-learn`. To save every projection figure at
full size, the 2-D coordinates (`*_projection.npz`) and the model comparisons:

```
python -c "from forensics_app.tools.feature_projection import export_all; export_all()"
```

Results are written to `outputs/set8_3_projection/` (one folder per model plus
`comparison/`); `outputs/` is git-ignored. The tool's dialog has the same
option.

## Course roadmap

The shell is intentionally neutral about the algorithms. Possible tool modules  
can follow the syllabus: filtering and edges, HOG, SIFT, bag of visual words,  
face detection/recognition, CNN classification and detection, transformers,  
semantic segmentation, and visualization/interpretability. Model weights and  
datasets should not be committed to the starter repository; document where they  
come from and keep paths configurable.

