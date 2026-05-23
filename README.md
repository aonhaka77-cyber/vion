# Vion

바이올린 악보 사진 위에 운지 번호를 표시하는 웹 편집기입니다.

- A현: 빨강
- D현: 파랑
- E현: 검정

프론트엔드는 정적 HTML이라 GitHub Pages로 호스팅할 수 있습니다. YOLO + Music Symbol Recognition 분석은 Python API 서버가 필요합니다.

## 파일 구조

```text
index.html              GitHub Pages 진입 파일
vion.html               실제 웹 앱
yolo_msr_server.py      YOLO/MSR 분석 API
requirements.txt        Python 서버 의존성
run_yolo_server.bat     Windows 로컬 서버 실행 파일
render.yaml             Render 배포 설정
Procfile                Python 웹 서버 실행 설정
```

## GitHub Pages 배포

1. GitHub 저장소를 만듭니다.
2. 이 폴더의 파일을 커밋해서 올립니다.
3. 저장소 `Settings > Pages`에서 배포 브랜치를 선택합니다.
4. 접속 주소는 보통 아래 형태입니다.

```text
https://사용자명.github.io/저장소명/
```

GitHub Pages는 Python 서버를 실행하지 않습니다. 그래서 YOLO/MSR API는 별도 호스팅이 필요합니다.

## 로컬 YOLO/MSR 서버 실행

Python 3.10 이상 설치 후:

```powershell
cd E:\dev
python -m pip install -r requirements.txt
python -m uvicorn yolo_msr_server:app --host 127.0.0.1 --port 8000
```

또는 Windows에서:

```powershell
.\run_yolo_server.bat
```

브라우저의 API 주소:

```text
http://localhost:8000/analyze-score
```

## 실제 YOLO 가중치 사용

학습된 YOLO 모델 파일이 있다면 환경변수로 지정합니다.

```powershell
$env:VION_YOLO_WEIGHTS="E:\dev\best.pt"
python -m uvicorn yolo_msr_server:app --host 127.0.0.1 --port 8000
```

`VION_YOLO_WEIGHTS`가 없거나 모델 로딩에 실패하면 OpenCV 기반 보조 탐지로 동작합니다.

## Render에 API 배포

1. GitHub 저장소를 Render에 연결합니다.
2. `render.yaml`을 사용해 Web Service를 생성합니다.
3. YOLO 가중치가 필요하면 Render 환경변수 `VION_YOLO_WEIGHTS`를 지정합니다.
4. 배포된 API 주소를 `vion.html`의 YOLO/MSR API 주소 입력칸에 넣습니다.

주의: GitHub Pages는 HTTPS이므로, 배포 API도 HTTPS 주소를 사용해야 브라우저에서 차단되지 않습니다.

## API 응답 형식

프론트엔드는 `notes`, `detections`, `symbols` 중 하나를 읽습니다.

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

좌표는 `bbox`, `x/y`, `cx/cy` 모두 지원합니다.
