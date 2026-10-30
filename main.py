import argparse
import csv
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

from config import VEHICLE_CLASSES, OCR_LANGUAGES, DEFAULT_PIXELS_PER_METER, SPEED_HISTORY


def parse_source(value):
    try:
        return int(value)
    except ValueError:
        return value


def plate_candidates(vehicle_crop):
    """Return likely rectangular license-plate regions from a vehicle crop."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return []

    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 80, 180)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    candidates = []

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 80:
            continue
        ratio = cw / max(ch, 1)
        area_ratio = area / max(w * h, 1)

        # Typical plate-like geometry; intentionally broad for different cameras.
        if 2.0 <= ratio <= 7.0 and 0.005 <= area_ratio <= 0.35:
            candidates.append((x, y, cw, ch))

    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    return candidates[:5]


def clean_plate_text(text):
    chars = []
    for ch in text.upper():
        if ch.isalnum():
            chars.append(ch)
    result = "".join(chars)
    return result if 4 <= len(result) <= 12 else ""


def recognize_plate(reader, vehicle_crop):
    best_text = ""
    best_conf = 0.0
    best_box = None

    for x, y, w, h in plate_candidates(vehicle_crop):
        plate = vehicle_crop[y:y+h, x:x+w]
        if plate.size == 0:
            continue

        # Upscale improves OCR on small plates.
        plate = cv2.resize(plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        result = reader.readtext(plate, detail=1, paragraph=False)

        for _, text, conf in result:
            cleaned = clean_plate_text(text)
            if cleaned and float(conf) > best_conf:
                best_text = cleaned
                best_conf = float(conf)
                best_box = (x, y, w, h)

    return best_text, best_conf, best_box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0 for webcam or path to video")
    parser.add_argument("--ppm", type=float, default=DEFAULT_PIXELS_PER_METER,
                        help="pixels-per-meter calibration")
    parser.add_argument("--speed-limit", type=float, default=100.0,
                        help="km/h alert threshold")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics vehicle model")
    args = parser.parse_args()

    source = parse_source(args.source)

    print("[1/4] Loading YOLO...")
    model = YOLO(args.model)

    print("[2/4] Loading EasyOCR...")
    reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)

    print("[3/4] Opening source...")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    os.makedirs("data/output", exist_ok=True)
    output_path = "data/output/annotated.mp4"
    csv_path = "data/output/detections.csv"

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # Track history: track_id -> deque[(time_seconds, center_x, center_y)]
    history = defaultdict(lambda: deque(maxlen=SPEED_HISTORY))
    plate_cache = {}
    last_ocr_time = defaultdict(float)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        log.writerow(["timestamp", "track_id", "vehicle", "speed_kmh", "plate", "ocr_confidence", "alert"])

        print("[4/4] LIVE DETECTION STARTED")
        print("Press Q to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False
            )

            annotated = frame.copy()

            if results and results[0].boxes is not None:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())

                    if boxes.id is not None:
                        track_id = int(boxes.id[i].item())
                    else:
                        track_id = -1

                    x1, y1, x2, y2 = xyxy
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    history[track_id].append((now, cx, cy))

                    speed_kmh = 0.0
                    if track_id >= 0 and len(history[track_id]) >= 2:
                        t0, x0, y0 = history[track_id][0]
                        t1, x1c, y1c = history[track_id][-1]
                        dt = max(t1 - t0, 1e-3)
                        pixel_distance = np.hypot(x1c - x0, y1c - y0)
                        meters = pixel_distance / max(args.ppm, 1e-6)
                        speed_kmh = (meters / dt) * 3.6

                    vehicle_name = VEHICLE_CLASSES.get(cls_id, "vehicle")
                    plate_text = plate_cache.get(track_id, ("", 0.0))[0]
                    plate_conf = plate_cache.get(track_id, ("", 0.0))[1]
                    plate_box = None

                    # OCR every ~1 second per tracked vehicle.
                    if track_id >= 0 and now - last_ocr_time[track_id] > 1.0:
                        crop = frame[y1:y2, x1:x2]
                        text, ocr_conf, rel_box = recognize_plate(reader, crop)
                        last_ocr_time[track_id] = now

                        if text:
                            plate_cache[track_id] = (text, ocr_conf)
                            plate_text, plate_conf = text, ocr_conf
                            plate_box = rel_box

                    alert = speed_kmh > args.speed_limit
                    label = f"{vehicle_name} ID:{track_id} {speed_kmh:.1f} km/h"
                    if plate_text:
                        label += f" | {plate_text}"

                    # Orange-ish default OpenCV BGR; no external styling dependency.
                    box_color = (0, 0, 255) if alert else (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(annotated, label, (x1, max(25, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                    if plate_box:
                        px, py, pw, ph = plate_box
                        ax1, ay1 = x1 + px, y1 + py
                        ax2, ay2 = ax1 + pw, ay1 + ph
                        cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (255, 255, 0), 2)

                    if alert:
                        cv2.putText(annotated, "SPEED ALERT", (x1, y2 + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                    log.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        track_id,
                        vehicle_name,
                        f"{speed_kmh:.2f}",
                        plate_text,
                        f"{plate_conf:.3f}",
                        int(alert)
                    ])

            cv2.putText(annotated, f"Calibration: {args.ppm:.2f} px/m",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"Limit: {args.speed_limit:.0f} km/h",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(annotated)
            cv2.imshow("AI Vehicle Speed + Number Plate Recognition", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\nFinished.")
    print(f"Video: {output_path}")
    print(f"CSV:   {csv_path}")


if __name__ == "__main__":
    main()
