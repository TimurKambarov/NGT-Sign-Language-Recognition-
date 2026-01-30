'''
Test real-time fingerspelling with streamlit UI
'''

import cv2
import numpy as np
import joblib
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
from collections import deque
import hand_face_detection as detection

# Configuration

MODEL_PATH = "models/random_forest_model.joblib"
LABEL_ENCODER_PATH = "models/label_encoder_rf.pkl"

SEQUENCE_LENGTH = 15
MIN_FRAMES = 5
PREDICTION_INTERVAL = 3
CONFIDENCE_THRESHOLD = 0.55

SIGNING_ZONE = {
    'x_min': -2.5,
    'x_max': 2.5,
    'y_min': -0.7,
    'y_max': 1.7,
}

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)
DISPLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Load Model

@st.cache_resource
def load_prediction_model():
    """Load trained Random Forest model and label encoder"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        return None, None
    
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, label_encoder

# Helper Classes and Functions

class UnifiedSequenceBuffer:
    def __init__(self, max_length=75):
        self.max_length = max_length
        self.frames_norm = deque(maxlen=max_length)
        self.frames_wrist_abs = deque(maxlen=max_length)
    
    def add_frame(self, landmarks_normalized, landmarks_absolute):
        frame_norm = []
        for lm in landmarks_normalized:
            frame_norm.extend([lm['x'], lm['y'], lm['z']])
        self.frames_norm.append(frame_norm)
        
        wrist_abs = landmarks_absolute[0]
        self.frames_wrist_abs.append([wrist_abs['x'], wrist_abs['y'], wrist_abs['z']])
    
    def clear(self):
        self.frames_norm.clear()
        self.frames_wrist_abs.clear()
    
    def is_ready(self, min_frames=30):
        return len(self.frames_norm) >= min_frames
    
    def __len__(self):
        return len(self.frames_norm)

def prepare_frame_features(landmarks_normalized, landmarks_absolute):
    features = []
    for lm in landmarks_normalized:
        features.extend([lm['x'], lm['y'], lm['z']])
    
    wrist_abs = landmarks_absolute[0]
    features.extend([wrist_abs['x'], wrist_abs['y'], wrist_abs['z']])
    
    return np.array(features).reshape(1, -1)

def is_in_signing_zone(relative_position):
    if relative_position is None:
        return False
    x, y = relative_position['rel_x'], relative_position['rel_y']
    return (SIGNING_ZONE['x_min'] <= x <= SIGNING_ZONE['x_max'] and
            SIGNING_ZONE['y_min'] <= y <= SIGNING_ZONE['y_max'])

# Video Processor

class VideoProcessor:
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder
        self.mp_resources = detection.initialize_mediapipe()
        self.sequence_buffer = UnifiedSequenceBuffer(max_length=SEQUENCE_LENGTH)
        
        self.mirrored = True
        self.show_zone = True
        self.frame_count = 0
        self.current_letter = None
        self.current_confidence = 0.0
        
        # FPS calculation
        self.fps = 0
        self.fps_counter = 0
        self.fps_start = cv2.getTickCount()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.mirrored:
            img = cv2.flip(img, 1)
        
        img, face_refs, hands_data = detection.process_frame(img, self.mp_resources)
        
        face_ok = face_refs is not None
        hand_ok = len(hands_data) > 0
        in_zone = False
        
        if hand_ok and face_ok:
            hand_data = hands_data[0]
            rel_pos = hand_data.get('relative_position')
            in_zone = is_in_signing_zone(rel_pos)
            
            if in_zone:
                self.sequence_buffer.add_frame(
                    hand_data['landmarks_normalized'],
                    hand_data['landmarks']
                )
                
                if self.sequence_buffer.is_ready(MIN_FRAMES) and self.frame_count % PREDICTION_INTERVAL == 0:
                    features = prepare_frame_features(hand_data['landmarks_normalized'], hand_data['landmarks'])
                    probabilities = self.model.predict_proba(features)[0]
                    idx = np.argmax(probabilities)
                    self.current_letter = self.label_encoder.inverse_transform([idx])[0]
                    self.current_confidence = probabilities[idx]
            else:
                self.sequence_buffer.clear()
                self.current_letter = None
                self.current_confidence = 0.0
        else:
            self.sequence_buffer.clear()
            self.current_letter = None
            self.current_confidence = 0.0

        # Drawing
        img = self.draw_annotations(img, face_refs, in_zone, hand_ok)
        
        self.frame_count += 1
        self.fps_counter += 1
        if self.fps_counter >= 30:
            fps_end = cv2.getTickCount()
            self.fps = 30 / ((fps_end - self.fps_start) / cv2.getTickFrequency())
            self.fps_start = fps_end
            self.fps_counter = 0
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def draw_annotations(self, frame, face_refs, in_zone, hand_ok):
        # Signing Zone
        if self.show_zone and face_refs:
            nose = face_refs['nose']
            fw, fh = face_refs['face_width'], face_refs['face_height']
            x1 = int(nose[0] + SIGNING_ZONE['x_min'] * fw)
            x2 = int(nose[0] + SIGNING_ZONE['x_max'] * fw)
            y1 = int(nose[1] + SIGNING_ZONE['y_min'] * fh)
            y2 = int(nose[1] + SIGNING_ZONE['y_max'] * fh)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_YELLOW, 2)

        # Prediction Box
        h, w, _ = frame.shape
        box_x, box_y, box_w, box_h = w - 220, 10, 210, 140
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_WHITE, 2)
        
        fill_pct = min(len(self.sequence_buffer) / SEQUENCE_LENGTH, 1.0)
        bar_width = int((box_w - 20) * fill_pct)
        cv2.rectangle(frame, (box_x + 10, box_y + box_h - 20), (box_x + 10 + bar_width, box_y + box_h - 10), COLOR_CYAN, -1)
        cv2.rectangle(frame, (box_x + 10, box_y + box_h - 20), (box_x + box_w - 10, box_y + box_h - 10), COLOR_WHITE, 1)

        if not in_zone:
            cv2.putText(frame, "MOVE HAND", (box_x + 35, box_y + 50), DISPLAY_FONT, 0.6, COLOR_RED, 2)
            cv2.putText(frame, "INTO ZONE", (box_x + 40, box_y + 75), DISPLAY_FONT, 0.6, COLOR_RED, 2)
        elif self.current_letter is None:
            cv2.putText(frame, "Detecting...", (box_x + 40, box_y + 60), DISPLAY_FONT, 0.6, COLOR_YELLOW, 2)
        elif self.current_confidence < CONFIDENCE_THRESHOLD:
            cv2.putText(frame, f"{self.current_letter}?", (box_x + 70, box_y + 65), DISPLAY_FONT, 1.8, COLOR_ORANGE, 3)
            cv2.putText(frame, f"{self.current_confidence:.0%}", (box_x + 80, box_y + 95), DISPLAY_FONT, 0.5, COLOR_ORANGE, 1)
        else:
            cv2.putText(frame, self.current_letter, (box_x + 70, box_y + 70), DISPLAY_FONT, 2.2, COLOR_GREEN, 4)
            cv2.putText(frame, f"{self.current_confidence:.0%}", (box_x + 80, box_y + 100), DISPLAY_FONT, 0.6, COLOR_GREEN, 2)

        # Status Bar
        cv2.rectangle(frame, (10, h - 35), (w - 10, h - 10), (0, 0, 0), -1)
        status = f"Face:{'OK' if face_refs else 'NO'} Hand:{'OK' if hand_ok else 'NO'} Zone:{'OK' if in_zone else 'NO'} FPS:{self.fps:.0f}"
        cv2.putText(frame, status, (15, h - 17), DISPLAY_FONT, 0.4, COLOR_WHITE, 1)
        
        return frame

# Streamlit UI

def main():
    st.set_page_config(page_title="NGT Sign Language Recognition", layout="wide")
    st.title("NGT Sign Language Recognition")
    st.markdown("""
    This application uses a Random Forest model to recognize Nederlandse Gebarentaal (NGT) finger-spelling in real-time.
    Ensure your face is visible and your hand is within the yellow signing zone.
    """)

    # Check for HTTPS (secure context required for camera access)
    import streamlit.components.v1 as components
    is_https = st.session_state.get('is_https', False)
    if not is_https:
        st.warning("this is a yellow rectangle")

    model, label_encoder = load_prediction_model()

    if model is None:
        st.error("Model files not found. Please ensure `models/random_forest_model.joblib` and `models/label_encoder_rf.pkl` exist.")
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Controls")
        mirrored = st.checkbox("Mirror View", value=True)
        show_zone = st.checkbox("Show Signing Zone", value=True)
        
        st.info("""
        **How to use:**
        1. Allow camera access when prompted by your browser.
        2. Position yourself so your face is detected.
        3. Perform signs in the yellow box.
        4. The prediction will appear in the top-right box.
        """)

    with col1:
        try:
            webrtc_ctx = webrtc_streamer(
                key="ngt-recognition",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: VideoProcessor(model, label_encoder),
                async_processing=True,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": True, "audio": False},
            )
        except Exception as e:
            st.error(f"Failed to start video stream. This may be due to camera permissions or the camera being in use by another application. Error: {e}")
            st.info("Please ensure your browser allows camera access and no other app is using the camera. Try refreshing the page or restarting your browser.")
            return

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.mirrored = mirrored
            webrtc_ctx.video_processor.show_zone = show_zone

if __name__ == "__main__":
    main()
