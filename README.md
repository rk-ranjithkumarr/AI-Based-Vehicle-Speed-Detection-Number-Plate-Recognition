# AI-Based Vehicle Speed Detection & Number Plate Recognition

Real-time computer-vision project using:
- YOLO + ByteTrack for vehicle detection/tracking
- OpenCV for video/webcam processing
- Perspective/calibration-friendly speed estimation
- OpenCV plate-region detection
- EasyOCR for license-plate text recognition
- CSV logging

## IMPORTANT ABOUT SPEED
The default project uses a configurable pixels-per-meter calibration. This is an estimate, not a legal speed-measuring instrument. For reliable real-world km/h, calibrate the camera using known road distances or a homography/perspective mapping.

## Run
1. Install Python 3.11.x.
2. Open Command Prompt in this folder.
3. Create a virtual environment:
   python -m venv .venv
4. Activate:
   .venv\Scripts\activate
5. Upgrade pip:
   python -m pip install --upgrade pip
6. Install libraries:
   pip install -r requirements.txt
7. Start webcam:
   python main.py --source 0
8. Or use a video:
   python main.py --source data/input/traffic.mp4

The first run downloads the YOLO model automatically.

## Useful options
python main.py --source 0 --ppm 8 --speed-limit 100
python main.py --source data/input/traffic.mp4 --ppm 8 --speed-limit 80

--ppm = approximate pixels per meter in the measurement direction.
Change it for your camera/scene after calibration.

Output:
- Annotated video: data/output/annotated.mp4
- Detection log: data/output/detections.csv

Press Q to stop live detection.
