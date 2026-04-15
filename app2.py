import streamlit as st
import cv2
import numpy as np
import tempfile 
from ultralytics import YOLO

st.set_page_config(page_title="Fog Vision System", layout="wide")
st.title(" Low Visibility Traffic Detection System")

# --- CACHE MODELS ---
@st.cache_resource
def load_models():
    model_traffic = YOLO('best.pt')
    model_vehicles = YOLO('yolov8n.pt')
    return model_traffic, model_vehicles

model_traffic, model_vehicles = load_models()
vehicle_classes = [2, 3, 5, 7] 

def apply_dehaze_pipeline(img, gamma_val, clahe_val, use_gamma=True):
    if use_gamma:
        invGamma = 1.0 / gamma_val
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
        
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clahe_val, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def verify_light_color(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    
    y1, y2 = max(0, y1-10), min(h, y2+10)
    x1, x2 = max(0, x1-10), min(w, x2+10)
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return "DETECTED" 
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    mask_red1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
    
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    
    if red_pixels > 3: return "RED"
    elif green_pixels > red_pixels and green_pixels > 5: return "GREEN"
    else: return "DETECTED" 
    
st.sidebar.header(" System Controls")

conf_light = st.sidebar.slider("Traffic Light Confidence", 0.1, 1.0, 0.20) 
conf_vehicle = st.sidebar.slider("Vehicle Confidence", 0.1, 1.0, 0.40)

st.sidebar.markdown("**2. Fog Removal Engine**")
use_dehaze = st.sidebar.checkbox(" Enable Dehazing Technique", value=True)
gamma_value = st.sidebar.slider("Gamma", 0.5, 3.0, 1.5, 0.1)
clahe_value = st.sidebar.slider("CLAHE", 1.0, 10.0, 3.5, 0.5)

st.sidebar.markdown("**3. Performance Tuning**")
frame_skip = st.sidebar.slider("Playback Speed (Frame Skip)", 1, 5, 2)

st.sidebar.markdown("**4. Video Input**")
uploaded_video = st.sidebar.file_uploader(" Browse and Select Video", type=['mp4', 'avi', 'mov'])

video_source = None
if uploaded_video is not None:
    # Save the uploaded file to a temporary location for OpenCV to read
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_source = tfile.name

start_button = st.sidebar.button(" Start Processing")
stop_button = st.sidebar.button(" Stop")

# --- MAIN DASHBOARD LAYOUT ---
status_placeholder = st.empty()

col1, col2 = st.columns(2)
with col1:
    st.subheader(" Raw Camera Feed")
    raw_placeholder = st.empty()
with col2:
    st.subheader(" Processed Output")
    processed_placeholder = st.empty()

st.markdown("---")
metrics_placeholder = st.empty()

if start_button:
    if video_source is None:
        st.error(" Please upload a video file from the sidebar first!")
    else:
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            st.error(" Cannot read the uploaded video file.")
            
        while cap.isOpened() and not stop_button:
            # Frame Skipping for performance
            for _ in range(frame_skip - 1):
                cap.grab() 
                
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
                continue
                
            frame = cv2.resize(frame, (800, 480))
            raw_display_frame = frame.copy()

            # 1.  DEHAZING
            if use_dehaze:
                processed_frame = apply_dehaze_pipeline(frame, gamma_value, clahe_value, use_gamma=True)
            else:
                processed_frame = frame.copy()

            # 2.  INFERENCE
            results_light = model_traffic(processed_frame, verbose=False, conf=conf_light)
            results_vehicle = model_vehicles(processed_frame, verbose=False, classes=vehicle_classes, conf=conf_vehicle)

            light_status = "SCANNING"
            obstacle_detected = False
            vehicles_count = 0
            
            for r in results_vehicle:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_name = model_vehicles.names[int(box.cls[0])]
                    
                    if cls_name == "truck": display_name = "Truck/Auto"
                    else: display_name = cls_name.capitalize()
                    
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(processed_frame, display_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    obstacle_detected = True
                    vehicles_count += 1

            for r in results_light:
                for box in r.boxes:
                    box_coords = list(map(int, box.xyxy[0]))
                    
                    verified_color = verify_light_color(raw_display_frame, box_coords) 
                    
                    if verified_color == "RED":
                        color = (0, 0, 255)
                        light_status = "RED"
                    elif verified_color == "GREEN":
                        color = (0, 255, 0)
                        light_status = "GREEN"
                    else:
                        color = (255, 0, 255)
                        light_status = "DETECTED"
                        
                    cv2.rectangle(processed_frame, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), color, 3)
                    cv2.putText(processed_frame, f"SIGNAL: {verified_color}", (box_coords[0], box_coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            raw_rgb = cv2.cvtColor(raw_display_frame, cv2.COLOR_BGR2RGB)
            processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            raw_placeholder.image(raw_rgb, channels="RGB", use_container_width=True)
            processed_placeholder.image(processed_rgb, channels="RGB", use_container_width=True)

            if light_status == "RED":
                status_placeholder.error(" **CRITICAL: RED SIGNAL DETECTED**")
            elif light_status == "GREEN" and obstacle_detected:
                status_placeholder.warning(" **WARNING: GREEN LIGHT but VEHICLES AHEAD**")
            elif light_status == "GREEN":
                status_placeholder.success(" **SAFE: GREEN SIGNAL & CLEAR PATH**")
            elif obstacle_detected:
                status_placeholder.info(" **CAUTION: LOW VISIBILITY - VEHICLES NEARBY**")
            else:
                status_placeholder.markdown(" **SYSTEM IDLE: Scanning environment...**")

            metrics_placeholder.markdown(f"""
            ###  Analytics
            * **Signal State:** {light_status}
            * **Vehicles Tracked:** {vehicles_count}
            * **FPS Skip:** {frame_skip}
            """)

        cap.release()