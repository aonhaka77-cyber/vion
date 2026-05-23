# Vion

Vion is a violin sheet-photo editor that writes fingering marks directly on top of an uploaded score image.

String colors:

- A string: red
- D string: blue
- E string: black

The frontend is static HTML and can be hosted on GitHub Pages. YOLO + Music Symbol Recognition requires a separate Python API server.

## Files

```text
index.html              GitHub Pages entry file
vion.html               Main web app
yolo_msr_server.py      YOLO/MSR analysis API
requirements.txt        Python server dependencies
run_yolo_server.bat     Local Windows server runner
render.yaml             Render deploy config
Procfile                Python web process command
runtime.txt             Python runtime hint
```

## GitHub Pages

1. Push this repository to GitHub.
2. Open repository Settings > Pages.
3. Select the `main` branch and root folder.
4. Open the Pages URL.

GitHub Pages only hosts the frontend. It cannot run the Python model server.

## Local API

Install Python 3.10 or newer, then run:

```powershell
cd E:\dev
python -m pip install -r requirements.txt
python -m uvicorn yolo_msr_server:app --host 127.0.0.1 --port 8000
```

Or run:

```powershell
.\run_yolo_server.bat
```

Use this API URL in the app:

```text
http://localhost:8000/analyze-score
```

## YOLO Weights

If you have a trained YOLO model, set:

```powershell
$env:VION_YOLO_WEIGHTS="E:\dev\best.pt"
python -m uvicorn yolo_msr_server:app --host 127.0.0.1 --port 8000
```

If `VION_YOLO_WEIGHTS` is not set or the model fails to load, the server falls back to an OpenCV-based detector.

## Deploy API On Render

1. Connect this GitHub repository to Render.
2. Create a Web Service using `render.yaml`.
3. If needed, set the `VION_YOLO_WEIGHTS` environment variable.
4. Put the deployed HTTPS API URL into the app's YOLO/MSR API input.

Important: GitHub Pages is HTTPS, so the API must also be HTTPS in production.

## API Response

The frontend accepts `notes`, `detections`, or `symbols`.

```json
{
  "notes": [
    {
      "bbox": [120, 300, 145, 325],
      "label": "notehead",
      "confidence": 0.92,
      "string": "A",
      "finger": "1"
    }
  ]
}
```

Supported coordinate formats:

- `bbox`: `[x1, y1, x2, y2]` or xywh with `bbox_format: "xywh"`
- `x` and `y`
- `cx` and `cy`
