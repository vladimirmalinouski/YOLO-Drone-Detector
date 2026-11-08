## Install dependencies

Run the commands below from the repository root (the folder containing
`detect-video.py` and `requirements.txt`). Use Python 3.10 or newer.

### 1. Install FFmpeg

The video detector requires both `ffmpeg` and `ffprobe` on your `PATH`.
These are system tools, installed separately from the Python dependencies.

On macOS with Homebrew installed:

```sh
brew install ffmpeg
```

On Ubuntu or Debian:

```sh
sudo apt update
sudo apt install ffmpeg python3-venv
```

Verify that both tools are available:

```sh
ffmpeg -version
ffprobe -version
```

### 2. Create and activate a Python virtual environment

On macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the Python requirements

With the virtual environment activated, download and install the packages
listed in the repository's `requirements.txt`:

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To leave the virtual environment when finished, run `deactivate`. In a new
terminal session, run `source .venv/bin/activate` again before using the detector.

## Run video detection on the test data

From the repository root, with the virtual environment activated, run:

```sh
python detect-video.py test-data/1.mp4 output/1-detected.mp4
```

The first argument is the input MP4; the second is the annotated output MP4.
The script creates the output folder if needed and draws red boxes around
detections. It loads `drone-detection-weights.pt` from the repository root, so
keep that file in place and run the command from that folder. Existing output
files at the specified path will be overwritten.

Try another included video:

```sh
python detect-video.py test-data/2.mp4 output/2-detected.mp4
```

The `test-data/` folder also contains `3.mp4` and `4.mp4`. Open the resulting
MP4 in your video player to review the detections.



## Data
The data used to train the YOLO model comes from two sources  
source #1  
[zenodo](https://zenodo.org/records/16017313?preview_file=datasets_seg.zip)  
this source has images of drones flying in the sky at a distance. the dataset has  
- 18061 RGB images
    - 10763 training
    - 2720 validation
    - 4578 for test


from this dataset I will use
- 2000 images for training
- 500 images for validation
- 500 images for test

! it is important to note that this dataset provides information in segmentation format, this segmentation data will need to be converted into


source #2
[kaggle](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset)
this source has images of drones flying close to the camera, which the other dataset doesn't have 

I will use all the data from this dataset which includes
- 1012 images for training
- 161 images validation 


with these two datasets the YOLO model will be trained with
- 3012 images for training
- 661 images for validation
- 500 images for test


### First Training Round Results (imgsz 640)
--- Test Results ---  
- Precision:  0.9307
- Recall:     0.8040
- mAP@50:     0.8716
- mAP@50-95:  0.4446

### Training after tuning hyperparameters and training for 150 epochs (imgsz 1280)
--- Test Results ---
- Precision:  0.9175
- Recall:     0.8980
- mAP@50:     0.9099
- mAP@50-95:  0.5102