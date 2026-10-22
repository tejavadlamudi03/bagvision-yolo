# BagVision YOLO – paper bag inspection

Sorts paper bags into **GOOD**, **BAD** or **EMPTY** from a top-down camera image, using a YOLO11 image classifier fine-tuned on your own photos.

The board can have several **bag places** (rectangles you mark once). The model looks at one place at a time, so it learns what a bag looks like, not where it lies.

## Setup

Python 3.10–3.12.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # webcam or saved photos
pip install -r requirements-realsense.txt               # + Intel RealSense camera (Linux/Windows)
```

The first training run downloads the YOLO base model (`yolo11s-cls.pt`, about 11 MB).

## Settings: `.env`

| Key | Default | Meaning |
|---|---|---|
| `CAMERA` | `realsense` | `realsense`, a webcam number (`0`, `1`, …) or a stream/video (`rtsp://…`, `video.mp4`) |
| `BAG_DATA_DIR` | `bag_data` | training photos and bag places |
| `BAG_MODEL_DIR` | `models` | trained model (`best.pt`) |
| `BAG_OUTPUT_DIR` | `inspections` | saved inspection results |
| `BASE_MODEL` | `yolo11s-cls.pt` | base for training (`yolo11n-cls.pt` is faster, `yolo11m-cls.pt` is bigger) |
| `IMGSZ` | `384` | image size the model sees |
| `DEVICE` | `auto` | `auto`, `cpu` or a GPU number |
| `BAD_THRESHOLD` | `0.5` | p(bad) needed to call a bag BAD (lower = stricter) |

The file holds no secrets. Values set in the shell override it, e.g. `CAMERA=0 python bag_inspect.py inspect`.

## Use

```bash
python bag_inspect.py capture          # 1. mark bag places, take labelled photos
python bag_inspect.py train            # 2. train (GPU: a few minutes)
python bag_inspect.py inspect          # 3. live: EMPTY / GOOD / BAD per bag place
python bag_inspect.py check PHOTOS/    # classify saved photos -> PHOTOS/bag_results.csv
```

**capture** keys:

| Key | Action |
|---|---|
| `1` / `2` | Drag a box around bag place 1 / 2, then press Enter |
| `N` | Board is empty: saves the reference picture. Press again if the light changes |
| `G` / `B` | The bag on the board is GOOD / BAD |
| `A` | Auto: one photo per second with the last label. Move and turn the bag meanwhile |
| `U` | Undo the last photo |
| `E` | Lock or unlock exposure (RealSense only) |
| `Q` | Quit |

Aim for at least 50 good, 30 bad and 30 empty photos. `train` needs 20 or more of each of good and bad. After training it prints a test score on photos the model never saw, including how many defects it missed. The previous model is kept in `models/history/`.

**check** uses the marked bag places if `bag_data/areas.json` exists. Otherwise each photo counts as one place.

## Docker

The Docker image is for training and for checking saved photos. It has no camera window.

```bash
docker build -t bagvision-yolo .                                   # CPU
docker build --build-arg TORCH=cu124 -t bagvision-yolo .           # NVIDIA GPU (run with --gpus all)

docker run --rm -v "$PWD/models:/app/models" -v "$PWD/photos:/photos" bagvision-yolo check /photos
docker run --rm -v "$PWD/models:/app/models" -v "$PWD/bag_data:/app/bag_data" bagvision-yolo train
```

## Files

```
bag_inspect.py              everything: capture, train, inspect, check
.env                        settings
requirements*.txt           Python packages
Dockerfile                  container for train / check
models/best.pt              trained model (created by `train`)
bag_data/                   your photos + bag places (not in git)
```
