import io
import os
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO
from utils import calculate_severity

try:
    model = YOLO("models/best.pt")
except Exception as e:
    model = None
    print(f"Warning: Could not load model 'models/best.pt'. {str(e)}")

def process_results(results, image_width, image_height):
    detections = []
    for result in results:
        for box in result.boxes:
            xywh = box.xywh[0].tolist()
            cls_id = int(box.cls[0].item())
            class_name = result.names[cls_id]
            confidence = float(box.conf[0].item())
            severity = calculate_severity(xywh, image_width, image_height)
            detections.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "bbox": [round(v, 2) for v in xywh],
                "severity": severity
            })
    return detections

def predict_image(image_bytes):
    if model is None:
        raise RuntimeError("Model 'models/best.pt' is not loaded or missing.")
        
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image_np.shape

    results = model(image_bgr, augment=True, conf=0.005, iou=0.1)
    return process_results(results, image_width, image_height)

def get_annotated_video(video_bytes, out_path):
    if model is None:
        raise RuntimeError("Model 'models/best.pt' is not loaded or missing.")

    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_in.write(video_bytes)
    temp_in.close()

    cap = cv2.VideoCapture(temp_in.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    best_detections = []
    track_history = {}
    alpha = 0.9  
    
    frame_count = 0
    frame_skip = 1
    last_results = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if frame_count % frame_skip == 0 or last_results is None:
            results = model.track(frame, persist=True, augment=True, conf=0.005, iou=0.1, tracker="botsort.yaml", verbose=False)
            last_results = results
        else:
            results = last_results
            
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
          
            has_ids = results[0].boxes.id is not None
            track_ids = results[0].boxes.id.int().cpu().numpy() if has_ids else [-1] * len(boxes)
            
            for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
                x1, y1, x2, y2 = box
                
                if track_id != -1:
                    if track_id in track_history:
                        px1, py1, px2, py2 = track_history[track_id]
                        x1 = alpha * x1 + (1 - alpha) * px1
                        y1 = alpha * y1 + (1 - alpha) * py1
                        x2 = alpha * x2 + (1 - alpha) * px2
                        y2 = alpha * y2 + (1 - alpha) * py2
                    track_history[track_id] = (x1, y1, x2, y2)
                
   
                w, h = x2 - x1, y2 - y1
                xywh = [x1 + w/2, y1 + h/2, w, h]
                severity = calculate_severity(xywh, width, height)
                
    
                color = (0, 255, 0) 
                if severity == "High":
                    color = (0, 0, 255) 
                elif severity == "Medium":
                    color = (0, 165, 255) 
                
         
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(annotated_frame, (ix1, iy1), (ix2, iy2), color, 2)
                
                class_name = results[0].names[int(cls_id)]
                id_str = f"ID: {track_id} | " if track_id != -1 else ""
                label = f"{id_str}{class_name} {conf:.2f} [{severity}]"
                
                
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (ix1, iy1 - 20), (ix1 + lw, iy1), color, -1)
                cv2.putText(annotated_frame, label, (ix1, iy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        frame_dets = process_results(results, width, height)
        
        if len(frame_dets) > len(best_detections):
            best_detections = frame_dets
            
        out.write(annotated_frame)

    cap.release()
    out.release()
    os.remove(temp_in.name)
    
    return best_detections
