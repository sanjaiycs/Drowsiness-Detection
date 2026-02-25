# Drowsiness Detection

This project detects driver drowsiness from a video file using MediaPipe
Face Landmarker and Eye Aspect Ratio (EAR).

------------------------------------------------------------------------

## Features

-   Face landmark detection using MediaPipe
-   Eye Aspect Ratio (EAR) computation
-   Drowsiness detection based on eye closure duration
-   Processes video input file
-   Saves processed output video with overlays

------------------------------------------------------------------------

## Project Structure

    .
    ├── mediapipe_drowsiness.py
    ├── face_landmarker.task
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## Installation

### 1. Create Virtual Environment (Recommended)

``` bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate    # Windows
```

### 2. Install Dependencies

``` bash
pip install -r requirements.txt
```

If installing manually:

``` bash
pip install mediapipe==0.10.20 opencv-python numpy
```

------------------------------------------------------------------------

## Download Face Landmarker Model

Download the MediaPipe face landmarker model:

``` bash
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

Place the downloaded file in the same directory as the Python script.

------------------------------------------------------------------------

## How to Run

Use the command line:

``` bash
python mediapipe_drowsiness.py input.mp4 output.mp4
```

Example:

``` bash
python mediapipe_drowsiness.py driver_video.mp4 result.mp4
```

-   input.mp4 → Your input video
-   output.mp4 → Processed output video with EAR and drowsiness status

------------------------------------------------------------------------

## How It Works

1.  Reads video frame-by-frame.
2.  Detects face landmarks using MediaPipe.
3.  Computes Eye Aspect Ratio (EAR).
4.  If EAR remains below threshold for a defined number of frames, the
    driver is marked as DROWSY.
5.  Saves the processed video with overlays.

------------------------------------------------------------------------

## Parameters You Can Adjust

Inside the script:

    EAR_THRESHOLD = 0.23
    FRAMES_THRESHOLD = 50

Lower EAR_THRESHOLD → More sensitive detection\
Higher FRAMES_THRESHOLD → Longer eye closure required to trigger alert

------------------------------------------------------------------------

## Requirements

-   Python 3.8+
-   mediapipe==0.10.20
-   opencv-python
-   numpy

------------------------------------------------------------------------

## Notes

-   Works best with clear frontal face videos.
-   Good lighting improves detection accuracy.
-   Designed for offline video processing (no frontend required).
