import os
import json
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from model import predict_image, get_annotated_video

app = FastAPI(title="Road Damage Severity Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/output", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
HISTORY_FILE = "history.json"

def save_history(record):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                history = json.load(f)
            except:
                pass
    history.append(record)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    latitude: float = Form(0.0),
    longitude: float = Form(0.0)
):
    try:
        file_bytes = await file.read()
        is_video = file.content_type and file.content_type.startswith('video')
        
        if is_video:
            filename = f"vid_{int(time.time())}.mp4"
            out_path = os.path.join("static", "output", filename)
            detections = get_annotated_video(file_bytes, out_path)
            video_url = f"/static/output/{filename}"
        else:
            detections = predict_image(file_bytes)
            video_url = None

        total_detections = len(detections)
        class_counts = {}
        severity_counts = {}
        has_high_severity = False
        
        for d in detections:
            cls = d['class']
            sev = d['severity']
            class_counts[cls] = class_counts.get(cls, 0) + 1
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            if sev == 'High':
                has_high_severity = True

        alert_msg = "🚨 ALERT: High severity damage detected! Immediate attention required." if has_high_severity else None
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "location": {"lat": latitude, "lng": longitude},
            "summary": {
                "total_detections": total_detections,
                "class_counts": class_counts,
                "severity_counts": severity_counts
            },
            "alert": alert_msg,
            "detections": detections,
            "video_url": video_url
        }
        
        save_history(record)
        return record
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/history")
async def get_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

@app.post("/repair/{timestamp}")
async def repair_damage(timestamp: str):
    if not os.path.exists(HISTORY_FILE):
        return {"status": "error"}
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    
    new_history = [r for r in history if r.get("timestamp") != timestamp]
    with open(HISTORY_FILE, "w") as f:
        json.dump(new_history, f, indent=4)
        
    return {"status": "success"}

@app.get("/report", response_class=HTMLResponse)
async def get_report(request: Request):
    if not os.path.exists(HISTORY_FILE):
        return templates.TemplateResponse(request=request, name="report.html", context={
            "request": request, "total_records": 0, "total_damages_found": 0, "severity_distribution": {}, "history": []
        })
    
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    
    total_damages = sum(r["summary"].get("total_detections", 0) for r in history)
    sev_dist = {"Low": 0, "Medium": 0, "High": 0}
    for r in history:
        for sev, count in r["summary"].get("severity_counts", {}).items():
            sev_dist[sev] = sev_dist.get(sev, 0) + count
            
    return templates.TemplateResponse(request=request, name="report.html", context={
        "request": request,
        "total_records": len(history),
        "total_damages_found": total_damages,
        "severity_distribution": sev_dist,
        "history": reversed(history)
    })

@app.post("/clear_history")
async def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    
    out_dir = os.path.join("static", "output")
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            filepath = os.path.join(out_dir, f)
            if os.path.isfile(filepath):
                os.remove(filepath)
                
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)
