# CNN & YOLO Modules Report

Overview of the CNN classification/projection tools (Sets 8.1–8.3) and YOLO object detection tools (Sets 9.1, 9.3), covering module responsibilities, preprocessing pipelines, and model parameters.

---

## 1. Module Functionalities

### CNN Modules
- **`forensics_app/tools/cnn_common.py`**: Shared PyTorch model definitions (`ShallowCNN`, `DeeperVGG`, `CompleteVGG`, `InceptionV3Classifier`, `ResNet50Classifier`), `CNN_ARCHITECTURES` registry, CUDA/CPU device selection (`get_compute_device`), canonical preprocessing (`build_preprocessing_spec`, `preprocess_pil_image`, `BatchPreprocessor`), dataset wrappers (`ImageRecordDataset`, `CIFAR10IndexedDataset`, `get_cifar10_test_cache`), and model/passport I/O (`save_cnn_bundle`, `load_cnn_bundle`, `load_imagenet_pretrained_bundle`).
- **`forensics_app/tools/cnn_trainer.py`** (*Set 8.1 — `CNNTrainerTool`*): Trains from-scratch or ImageNet-pretrained transfer-learning CNNs on Image Folder or CIFAR-10 datasets using CUDA AMP (`torch.amp`), `cudnn.benchmark`, GPU batch preprocessing, optional frozen-backbone feature caching, and graceful early stopping (`Stop Training`). Evaluates on held-out test data and saves `weights.pt`, `passport.json`, and `training_curves.png`.
- **`forensics_app/tools/cnn_classifier.py`** (*Set 8.2 — `CNNClassificationTool`*): Loads a saved CNN bundle or an off-the-shelf ImageNet-1K model (`resnet50` / `inception_v3`), displays the model passport and `training_curves.png`, and runs inference on an individual held-out test image, a held-out test group (with confusion matrix), or a new canvas/disk image.
- **`forensics_app/tools/cnn_feature_projection.py`** (*Set 8.3 — `CNNFeatureProjectionTool`*): Extracts deep feature vectors (`fc1`, `custom_dense`, or `avgpool`) from the test set (saved to `features_test.npz` + `features_meta.json`), reloads saved `.npz` archives, optionally applies PCA pre-reduction (`> 50` dims), projects vectors to 2D via t-SNE and/or UMAP, computes silhouette scores, and renders `projection_2d.png`.

### YOLO Modules
- **`forensics_app/tools/yolo_detection.py`** (*Set 9.1 — `YoloDetectionTool`*): Runs Ultralytics YOLO object detection on the active canvas image (defaulting to COCO-pretrained `yolo11n.pt`, auto-downloaded to `models/yolo/pretrained/` on first use). Supports custom `.pt` weights, confidence/IoU/input-size/device/class filters, LRU model caching (`MODEL_CACHE_SIZE = 3`), and color-coded bounding-box rendering.
- **`forensics_app/tools/blood_cell_detection.py`** (*Set 9.3 — `BloodCellDetectionTool`*): Evaluates exported blood-cell YOLO11n models (`models/yolo/<model>/` with `best.pt` + `passport.json`) against YOLO `.txt` ground-truth labels for platelets (`PC`), red blood cells (`RBC`), and white blood cells (`WBC`). Supports single-image overlays (TP/FP/FN/border-cut breakdown), full-folder evaluation with CSV export, and side-by-side comparison of fine-tuned vs. from-scratch models.

---

## 2. Preprocessing Pipelines

### CNN Preprocessing (`cnn_common.py`, `cnn_trainer.py`, `cnn_classifier.py`, `cnn_feature_projection.py`)
1. **Color Conversion**: Input PIL image is converted to `RGB` (`image.convert("RGB")`).
2. **Spatial Resizing**: Bilinear resize to `[input_size, input_size]` (`Image.Resampling.BILINEAR` on CPU, or `torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)` for vectorized/GPU batches).
3. **Pixel Scaling & Normalization**:
   - `unit_0_1` (from-scratch CNNs): Scales `uint8` pixels `[0, 255]` → `float32 [0.0, 1.0]` (`x / 255.0`; `mean=[0.0, 0.0, 0.0]`, `std=[1.0, 1.0, 1.0]`).
   - `imagenet` (transfer learning & ImageNet-1K): Scales `[0, 255]` → `[0.0, 1.0]`, then channel-standardizes using `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`.
4. **Tensor Layout**: Arranged as contiguous `NCHW` (`(B, 3, H, W)`).
5. **Caching & GPU Acceleration**:
   - `CIFAR10IndexedDataset` preloads normalized `float32` tensors when estimated size ≤ 768 MB (e.g., `32×32`), or retains native `32×32` `uint8` tensors in RAM (~138 MB for 45k images) and delegates resizing/normalization to `BatchPreprocessor` on CUDA.
   - `ImageRecordDataset` caches decoded/preprocessed tensors in RAM after the first pass when estimated size ≤ 512 MB.
   - `CNNFeatureProjectionTool` reduces features > 50-D to `min(50, N - 1)` via PCA before t-SNE/UMAP when `N > 50`.

### YOLO Preprocessing (`yolo_detection.py`, `blood_cell_detection.py`, exported passports)
1. **Inference Input (`detect_objects`)**:
   - Converts PIL image to `RGB` (`image.convert("RGB")`), then flips channels to contiguous `BGR` (`np.ascontiguousarray(np.asarray(rgb)[..., ::-1])`) to match Ultralytics/OpenCV channel order.
   - Passes `imgsz` (dialog clamps to multiples of `32`, default `640`) to `model.predict(...)`; Ultralytics internally letterboxes the image to `imgsz × imgsz` and scales pixel values to `[0.0, 1.0]`.
2. **Blood-Cell Dataset & Training Preprocessing (per exported `passport.json`)**:
   - **Offline Roboflow export**: Auto-orient and stretch-resize to `640×640`. Train split (`3,027` images) generated from `1,009` source images × 3 augmentations (flips, 90° rotations, blur, salt-and-pepper noise); validation (`288` images) and test (`144` images) splits are un-augmented.
   - **Online Ultralytics training augmentations**: `imgsz=640`, `mosaic=1.0` (`close_mosaic=10`), `fliplr=0.5`, `flipud=0.5`, `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `translate=0.1`, `scale=0.5`.

---

## 3. Model Architectures & Parameters

### CNN Architectures (`CNN_ARCHITECTURES` & ImageNet-1K Bundles)

| `arch_key` | Architecture | Default / Min Input | Normalization | Default Optimizer & LR | Feature Layer (Dim) | Parameter Count (Example) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `shallow_cnn` | 1× Conv32 → MaxPool → Dense100 | `32` / `16` | `unit_0_1` | `SGD` (`0.01`) | `fc1` (`100`) | Depends on `input_size` & `num_classes` |
| `deeper_vgg` | 2× Conv32 → MaxPool → Dense128 | `32` / `16` | `unit_0_1` | `SGD` (`0.01`) | `fc1` (`128`) | `1,060,138` total (10 classes, `32×32`) |
| `complete_vgg` | 2× Conv64 → 2× Conv128 → 2× Conv256 → Dense128 | `64` / `32` | `unit_0_1` | `SGD` (`0.01`) | `fc1` (`128`) | Depends on `input_size` & `num_classes` |
| `inception_v3` | InceptionV3 backbone → Dense1024 → Dropout(0.2) | `299` / `75` | `imagenet` | `Adam` (`0.001`) | `custom_dense` (`1024`) / `avgpool` (`2048`) | `23,893,994` total / `2,108,426` trainable (10 classes, frozen backbone) |
| `resnet50` | ResNet50 backbone → Linear classifier | `224` / `32` | `imagenet` | `Adam` (`0.001`) | `avgpool` (`2048`) | `23.5M+` backbone + `2048 × num_classes` head |

- **CNN Trainer (`train_cnn_model` / UI) Key Parameters**:
  - `epochs`: `15` (code default), UI sets `20` (from scratch) or `8` (transfer learning); `fine_tune_epochs`: `0` default (`fine_tune_lr=1e-4`).
  - `batch_size`: `128` (UI default; capped at `64` on `<= 4096` MB VRAM when `input_size >= 224`), `32` (programmatic default).
  - `test_size`: `0.2`, `val_split`: `0.1`, `random_state`: `42`, `device`: `"auto"` (`cuda:0` when available).
  - Mixed precision (`torch.amp.autocast("cuda")` + `GradScaler`) and `cudnn.benchmark` enabled on CUDA; frozen-backbone feature caching used for Stage-1 transfer learning.
- **CNN Feature Projection (`project_features_2d`) Key Parameters**:
  - `batch_size=64` (capped at `64` on `<= 4096` MB VRAM for `input_size >= 224`), `method="both"` (`tsne` / `umap` / `both`), `perplexity=30.0`, `tsne_max_iter=500`, `umap_n_neighbors=15`, `umap_min_dist=0.1`, `umap_metric="euclidean"`, `random_state=42`.

### YOLO Models (`yolo_detection.py` & Exported Blood-Cell Models)

| Model | Weights File | Task / Classes | Layers | Parameters | GFLOPs | Inference Defaults | Training / Evaluation Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11n (Pretrained)** | `models/yolo/pretrained/yolo11n.pt` | COCO Object Detection (`80` classes) | — | — | — | `imgsz=640`, `conf=0.25`, `iou=0.7`, `device="auto"` | Downloaded automatically by Ultralytics on first use |
| **`blood_cell_yolo11n`** | `models/yolo/blood_cell_yolo11n/best.pt` | Blood Cells (`PC`, `RBC`, `WBC` — 3 classes) | `182` | `2,605,369` | `6.6` | `imgsz=640`, `conf=0.25`, `iou=0.7` | Fine-tuned from `yolo11n.pt` (50 epochs, `AdamW` `lr=0.001429`, `batch=16`); test `mAP50=0.988`, `mAP50-95=0.855` |
| **`blood_cell_yolo11n_scratch`** | `models/yolo/blood_cell_yolo11n_scratch/best.pt` | Blood Cells (`PC`, `RBC`, `WBC` — 3 classes) | `181` | `2,590,425` | `6.5` | `imgsz=640`, `conf=0.25`, `iou=0.7` | Trained from `yolo11n.yaml` (30 epochs, `AdamW` `lr=0.001429`, `batch=16`); test `mAP50=0.983`, `mAP50-95=0.832` |
