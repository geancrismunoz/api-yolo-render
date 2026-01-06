from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import numpy as np
import cv2
import uuid
import os

# ---------------- CONFIG ----------------
MAX_IMAGE_MB = 5
IMG_SIZE = 640
CONFIDENCE = 0.5
# ---------------------------------------

app = FastAPI()

# cargar modelo UNA sola vez
model = YOLO("models/yolov8l-seg.pt")

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.post("/segmentar")
async def segmentar_imagen(file: UploadFile = File(...)):

    # validar tamaño
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_IMAGE_MB:
        raise HTTPException(status_code=400, detail="Imagen demasiado grande")

    # leer imagen
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Imagen inválida")

    # resize para CPU
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # inferencia
    results = model(img, conf=CONFIDENCE)
    r = results[0]

    # imagen segmentada
    img_seg = r.plot()

    # guardar imagen
    nombre = f"{uuid.uuid4()}.jpg"
    ruta = f"outputs/{nombre}"
    cv2.imwrite(ruta, img_seg)

    # detecciones
    detecciones = []
    if r.boxes is not None:
        for box in r.boxes:
            detecciones.append({
                "clase": model.names[int(box.cls)],
                "confianza": round(float(box.conf), 3)
            })

    return JSONResponse({
        "total_detecciones": len(detecciones),
        "detecciones": detecciones,
        "imagen_segmentada_url": f"/outputs/{nombre}"
    })
