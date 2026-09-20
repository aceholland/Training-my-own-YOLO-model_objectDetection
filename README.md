# Training-my-own-YOLO-model_objectDetection
# Room Object Detector (YOLO26)

Custom-trained YOLO26 model for detecting everyday indoor objects — people, furniture, electronics, and personal items — fine-tuned on a subset of the [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) dataset.

## Overview

This project fine-tunes Ultralytics' YOLO26s model on a curated 38-class subset of Open Images V7, focused on common household and personal objects, trained in Google Colab and deployed locally via Anaconda.

## Classes Detected

- **People/Body**: Person, Human face, Human hand, Human head, Human eye, Human arm, Human leg
- **Furniture**: Bed, Chair, Table, Sofa, Desk, Cabinetry, Bookcase, Wardrobe
- **Everyday objects**: Bottle, Cup, Mug, Bowl, Fork, Knife, Spoon, Plate, Book
- **Electronics**: Laptop, Mobile phone, Computer keyboard, Computer mouse, Television, Remote control
- **Food**: Coffee cup
- **Clothing**: Shirt, Jacket, Shoe, Hat, Glasses, Backpack
- **Misc**: Toy, Curtain

## Results

| Metric | Score |
|---|---|
| mAP50 | 0.886 |
| mAP50-95 | 0.738 |
| Precision | 0.817 |
| Recall | 0.828 |

Trained for 50 epochs at 640x640 resolution on a Tesla T4 GPU (Google Colab), starting from the `yolo26s.pt` COCO-pretrained checkpoint.

Some classes (e.g. Remote control, Bookcase) had very few training examples and perform poorly — see [Known Limitations](#known-limitations).

---

## Step 1 — Download the Dataset (Google Colab)

Install dependencies (Pillow version pin avoids a known conflict with `fiftyone`):

```python
!pip uninstall -y pillow -q
!pip install "pillow==11.0.0" fiftyone ultralytics -q
```

**Restart the Colab runtime** after this step (Runtime → Restart session), then continue.

Download a 300-image subset of Open Images V7 containing only the classes above, and export it in YOLO format:

```python
import fiftyone as fo
import fiftyone.zoo as foz

classes = [
    "Person", "Human face", "Human hand", "Human head", "Human eye", "Human arm", "Human leg",
    "Bed", "Chair", "Table", "Sofa", "Desk", "Cabinetry", "Bookcase", "Wardrobe",
    "Bottle", "Cup", "Mug", "Bowl", "Fork", "Knife", "Spoon", "Plate", "Book",
    "Laptop", "Mobile phone", "Computer keyboard", "Computer mouse", "Television", "Remote control",
    "Coffee cup",
    "Shirt", "Jacket", "Shoe", "Hat", "Glasses", "Backpack",
    "Toy", "Curtain"
]

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=classes,
    max_samples=300,
)

dataset.export(
    export_dir="/content/dataset",
    dataset_type=fo.types.YOLOv5Dataset,
    classes=classes,
)
```

This creates `/content/dataset/` containing `images/`, `labels/`, and a starter `dataset.yaml`.

## Step 2 — Fix the Dataset YAML

FiftyOne's export doesn't always include the required `train`/`val` keys, and class counts can mismatch. Rebuild the yaml manually:

```python
names_list = ['Person','Human face','Human hand','Human head','Human eye','Human arm','Human leg',
'Bed','Chair','Table','Sofa','Desk','Cabinetry','Bookcase','Wardrobe','Bottle','Cup','Mug','Bowl',
'Fork','Knife','Spoon','Plate','Book','Laptop','Mobile phone','Computer keyboard','Computer mouse',
'Television','Remote control','Coffee cup','Shirt','Jacket','Shoe','Hat','Glasses','Backpack','Toy','Curtain']

yaml_content = f"""
path: /content/dataset
train: images
val: images
nc: {len(names_list)}
names: {names_list}
"""
with open("/content/dataset/dataset.yaml", "w") as f:
    f.write(yaml_content)
```

## Step 3 — Train the Model (Google Colab)

Make sure Colab's runtime is set to a **GPU** (Runtime → Change runtime type → T4 GPU), then run:

```bash
!pip install -U ultralytics -q
!yolo detect train data=/content/dataset/dataset.yaml model=yolo26s.pt epochs=50 imgsz=640
```

Training results and weights are saved to `/content/runs/detect/train/` (or `train-2`, `train-3`, etc. if re-run). The best-performing weights are saved as `weights/best.pt`.

## Step 4 — Test the Model

Run inference on the validation images to sanity-check detections:

```bash
!yolo detect predict model=/content/runs/detect/train/weights/best.pt source=/content/dataset/images/val save=True
```

Display a sample of the results inline:

```python
import glob
from IPython.display import Image, display

for image_path in glob.glob('/content/runs/detect/predict/*.jpg')[:10]:
    display(Image(filename=image_path, height=400))
    print('\n')
```

## Step 5 — Download the Trained Model

Package the weights and training logs into a zip file:

```python
!mkdir /content/my_model
!cp /content/runs/detect/train/weights/best.pt /content/my_model/my_model.pt
!cp -r /content/runs/detect/train /content/my_model

%cd my_model
!zip /content/my_model.zip my_model.pt
!zip -r /content/my_model.zip train
%cd /content
```

Download it to your computer:

```python
from google.colab import files
files.download('/content/my_model.zip')
```

Unzip it locally — you should have `my_model.pt` plus the training logs folder.

## Step 6 — Deploy Locally (Anaconda)

Set up a dedicated environment:

```bash
conda create -n yolo-env1 python=3.10 -y
conda activate yolo-env1

pip install ultralytics
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Download the inference helper script:

```bash
curl -o yolo_detect.py https://www.ejtech.io/code/yolo_detect.py
```

Place `my_model.pt` in the same folder as `yolo_detect.py`, then run detection:

**Webcam:**
```bash
python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720
```

**Image, video, or folder:**
```bash
python yolo_detect.py --model my_model.pt --source path/to/image_or_video --thresh 0.5
```

### Script arguments

| Argument | Description | Default |
|---|---|---|
| `--model` | Path to your trained `.pt` file | required |
| `--source` | Image, folder, video file, or `usb0`/`picamera0` for a live camera | required |
| `--thresh` | Minimum confidence to display a detection | 0.5 |
| `--resolution` | Display resolution, e.g. `1280x720` | source resolution |
| `--record` | Save output as `demo1.avi` (requires `--resolution`) | off |

---

## Known Limitations

- Detects object **categories** only (e.g. "a bottle"), not the identity of a specific personal item.
- Classes with very few training examples (Remote control, Bookcase) detect unreliably — more images per class are needed to fix this.
- No dedicated Open Images classes exist for: pen, pen stand, papers, notebook, or toy car. Detecting these would require collecting and manually labeling your own photos (e.g. with Roboflow or LabelImg), then merging that data with the Open Images set before training.
- Increasing `max_samples` per class in Step 1, or downloading extra samples specifically for weak classes, improves accuracy at the cost of longer download/training time.

## Credits

This project follows the workflow taught in **"How to Train YOLO Object Detection Models in Google Colab (YOLO26, YOLO11, YOLOv8)"** by [EJ Technology Consultants](https://www.ejtech.io) — watch it here: https://youtu.be/r0RspiLG260

The `yolo_detect.py` inference script used for local deployment is also provided by EJ Technology Consultants ([ejtech.io](https://www.ejtech.io)).

## License

This project uses Ultralytics YOLO, licensed under AGPL-3.0. See [Ultralytics licensing](https://www.ultralytics.com/license) for commercial use terms.
