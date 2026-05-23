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


def label_to_string(label: str, center_y: float, image_h: int, staff: dict[str, Any] | None = None) -> str:
    text = label.upper()
    if "A" in text:
        return "A"
    if "D" in text:
        return "D"
    if "E" in text:
        return "E"
    if staff:
        relative = (center_y - staff["top"]) / max(1.0, staff["bottom"] - staff["top"])
        if relative < 0.20:
            return "E"
        if relative < 0.67:
            return "A"
        return "D"
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


def find_staff_groups(gray: np.ndarray) -> list[dict[str, Any]]:
    h, w = gray.shape
    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(45, w // 18), 1))
    lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    row_counts = np.count_nonzero(lines_img, axis=1)
    threshold = max(30, w * 0.16)

    lines = []
    start = None
    best_y = 0
    best_count = 0
    for y, count in enumerate(row_counts):
        if y < h * 0.10:
            continue
        if count > threshold:
            if start is None:
                start = y
                best_y = y
                best_count = count
            elif count > best_count:
                best_y = y
                best_count = count
        elif start is not None:
            lines.append(best_y)
            start = None
            best_count = 0
    if start is not None:
        lines.append(best_y)

    compact = []
    for line in lines:
        if not compact or line - compact[-1] > 4:
            compact.append(line)

    groups = []
    i = 0
    while i <= len(compact) - 5:
        slice_ = compact[i : i + 5]
        gaps = [slice_[idx + 1] - slice_[idx] for idx in range(4)]
        avg_gap = sum(gaps) / len(gaps)
        stable = all(abs(gap - avg_gap) < avg_gap * 0.42 for gap in gaps)
        if stable and 5 <= avg_gap <= 42:
            groups.append(
                {
                    "lines": slice_,
                    "top": float(slice_[0]),
                    "bottom": float(slice_[4]),
                    "center": float((slice_[0] + slice_[4]) / 2),
                    "gap": float(avg_gap),
                }
            )
            i += 5
        else:
            i += 1
    return groups


def nearest_staff(staff_groups: list[dict[str, Any]], y: float) -> dict[str, Any] | None:
    if not staff_groups:
        return None
    return min(staff_groups, key=lambda staff: abs(staff["center"] - y))


def detect_with_yolo(image: np.ndarray, level: str) -> list[dict[str, Any]]:
    model = load_yolo_model()
    if model is None:
        return []

    result = model.predict(image, verbose=False, conf=0.2)[0]
    names = result.names
    detections = []
    h, _w = image.shape[:2]
    staff_groups = find_staff_groups(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    for idx, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        label = str(names.get(cls_id, cls_id)).lower()
        if not any(token in label for token in ["note", "head", "quarter", "eighth", "half", "whole"]):
            continue

        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        string_name = label_to_string(label, cy, h, nearest_staff(staff_groups, cy))
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

    return sorted(detections, key=lambda item: (item.get("staff_index", round(center(item)[1] / 45)), center(item)[0]))


def detect_with_opencv(image: np.ndarray, level: str) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    staff_groups = find_staff_groups(gray)
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
    for staff in staff_groups:
        for line_y in staff["lines"]:
            y1 = max(0, int(line_y) - 2)
            y2 = min(h, int(line_y) + 3)
            no_lines[y1:y2, :] = 0

    note_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    notes = cv2.morphologyEx(no_lines, cv2.MORPH_CLOSE, note_kernel)
    contours, _ = cv2.findContours(notes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    min_area = max(10, (w * h) * 0.000006)
    max_area = (w * h) * 0.0016

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        cx = x + bw / 2
        cy = y + bh / 2
        staff = nearest_staff(staff_groups, cy)
        if staff is None:
            continue
        gap = staff["gap"]
        in_violin_staff_zone = staff["top"] - gap * 3.3 <= cy <= staff["bottom"] + gap * 3.1
        if not in_violin_staff_zone:
            continue
        if area < min_area or area > max_area:
            continue
        if x < w * 0.07 or y < h * 0.12:
            continue
        ratio = bw / max(1, bh)
        if ratio < 0.35 or ratio > 2.8:
            continue
        if bw < gap * 0.35 or bh < gap * 0.30:
            continue
        if bw > gap * 3.2 or bh > gap * 3.6:
            continue

        candidates.append((x, y, bw, bh, cx, cy, area, staff_groups.index(staff), staff))

    candidates = dedupe_candidates(candidates, min_distance=max(12, min(w, h) * 0.014))
    detections = []
    for idx, (x, y, bw, bh, cx, cy, _area, staff_index, staff) in enumerate(candidates[:120]):
        string_name = label_to_string("", cy, h, staff)
        detections.append(
            {
                "bbox": [x, y, x + bw, y + bh],
                "label": "notehead",
                "confidence": 0.45,
                "string": string_name,
                "finger": finger_for(idx, level, string_name),
                "staff_index": staff_index,
            }
        )

    return sort_detections(detections)


def dedupe_candidates(candidates: list[tuple], min_distance: float) -> list[tuple]:
    picked = []
    for candidate in sorted(candidates, key=lambda item: item[6], reverse=True):
        duplicate = False
        for existing in picked:
            if ((candidate[4] - existing[4]) ** 2 + (candidate[5] - existing[5]) ** 2) ** 0.5 < min_distance:
                duplicate = True
                break
        if not duplicate:
            picked.append(candidate)
    return sorted(picked, key=lambda item: (item[7], item[4]))


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
