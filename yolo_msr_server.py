import io
import os
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image


app = FastAPI(title="Vion YOLO/MSR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YOLO_MODEL = None
YOLO_LOAD_ERROR = ""


def load_yolo_model() -> Any:
    global YOLO_MODEL, YOLO_LOAD_ERROR
    if YOLO_MODEL is not None:
        return YOLO_MODEL

    weights = os.getenv("VION_YOLO_WEIGHTS", "").strip()
    if not weights:
        YOLO_LOAD_ERROR = "VION_YOLO_WEIGHTS is not set"
        return None

    try:
        from ultralytics import YOLO

        YOLO_MODEL = YOLO(weights)
        YOLO_LOAD_ERROR = ""
        return YOLO_MODEL
    except Exception as exc:
        YOLO_LOAD_ERROR = str(exc)
        return None


def read_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def label_to_string(label: str, center_y: float, image_h: int) -> str:
    text = label.upper()
    if "A" in text:
        return "A"
    if "D" in text:
        return "D"
    if "E" in text:
        return "E"
    if center_y < image_h * 0.38:
        return "E"
    if center_y < image_h * 0.68:
        return "A"
    return "D"


def finger_for(index: int, level: str, string_name: str) -> str:
    patterns = {
        "beginner": ["0", "1", "2", "3", "1", "2", "0", "1", "3", "4"],
        "middle": ["1", "2", "3", "1", "4", "2", "3", "1", "2", "4"],
        "advanced": ["2", "3", "4", "1", "3", "2", "4", "2", "1", "3"],
    }
    pattern = patterns.get(level, patterns["middle"])
    if string_name == "E" and index % 5 == 0:
        return "4"
    if string_name == "D" and index % 6 == 0:
        return "0"
    return pattern[index % len(pattern)]


def detect_with_yolo(image: np.ndarray, level: str) -> list[dict[str, Any]]:
    model = load_yolo_model()
    if model is None:
        return []

    result = model.predict(image, verbose=False, conf=0.2)[0]
    names = result.names
    detections = []
    h, _w = image.shape[:2]

    for idx, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        label = str(names.get(cls_id, cls_id)).lower()
        if not any(token in label for token in ["note", "head", "quarter", "eighth", "half", "whole"]):
            continue

        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        string_name = label_to_string(label, cy, h)
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "confidence": float(box.conf[0]),
                "string": string_name,
                "finger": finger_for(idx, level, string_name),
            }
        )

    return sort_detections(detections)


def sort_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def center(item: dict[str, Any]) -> tuple[float, float]:
        x1, y1, x2, y2 = item["bbox"]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    return sorted(detections, key=lambda item: (round(center(item)[1] / 45), center(item)[0]))


def detect_with_opencv(image: np.ndarray, level: str) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )

    horizontal = binary.copy()
    kernel_w = max(30, w // 24)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    staff_lines = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, line_kernel)
    no_lines = cv2.subtract(binary, staff_lines)

    note_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    notes = cv2.morphologyEx(no_lines, cv2.MORPH_CLOSE, note_kernel)
    contours, _ = cv2.findContours(notes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    min_area = max(12, (w * h) * 0.000008)
    max_area = (w * h) * 0.004

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        if x < w * 0.07 or y < h * 0.12:
            continue
        ratio = bw / max(1, bh)
        if ratio < 0.25 or ratio > 3.2:
            continue
        if bw > w * 0.07 or bh > h * 0.09:
            continue

        cx = x + bw / 2
        cy = y + bh / 2
        candidates.append((x, y, bw, bh, cx, cy, area))

    candidates = dedupe_candidates(candidates, min_distance=max(14, min(w, h) * 0.018))
    detections = []
    for idx, (x, y, bw, bh, cx, cy, _area) in enumerate(candidates[:120]):
        string_name = label_to_string("", cy, h)
        detections.append(
            {
                "bbox": [x, y, x + bw, y + bh],
                "label": "notehead",
                "confidence": 0.45,
                "string": string_name,
                "finger": finger_for(idx, level, string_name),
            }
        )

    return sort_detections(detections)


def dedupe_candidates(candidates: list[tuple], min_distance: float) -> list[tuple]:
    picked = []
    for candidate in sorted(candidates, key=lambda item: item[-1], reverse=True):
        duplicate = False
        for existing in picked:
            if ((candidate[4] - existing[4]) ** 2 + (candidate[5] - existing[5]) ** 2) ** 0.5 < min_distance:
                duplicate = True
                break
        if not duplicate:
            picked.append(candidate)
    return sorted(picked, key=lambda item: (round(item[5] / 45), item[4]))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "yolo_loaded": load_yolo_model() is not None,
        "yolo_error": YOLO_LOAD_ERROR,
    }


@app.post("/analyze-score")
async def analyze_score(
    image: UploadFile = File(...),
    level: str = Form("middle"),
) -> JSONResponse:
    file_bytes = await image.read()
    cv_image = read_image(file_bytes)

    detections = detect_with_yolo(cv_image, level)
    engine = "yolo"
    if not detections:
        detections = detect_with_opencv(cv_image, level)
        engine = "opencv-fallback"

    h, w = cv_image.shape[:2]
    return JSONResponse(
        {
            "engine": engine,
            "image_width": w,
            "image_height": h,
            "notes": detections,
            "yolo_error": YOLO_LOAD_ERROR,
        }
    )

