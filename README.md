https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset
https://zenodo.org/records/16017313?preview_file=datasets_seg.zip


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
