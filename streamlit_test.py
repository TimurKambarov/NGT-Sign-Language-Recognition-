"""
NGT Sign Language Recognition - Streamlit Web Interface
Real-time recognition using Random Forest for static gestures

Features:
- Real-time video with streamlit-webrtc
- Random Forest model for 23 static letters
- Image-based guides from data/NGT_Gestures/

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import cv2
import numpy as np
import joblib
import os
from collections import deque
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import hand_face_detection as detection


# Configuration

MODEL_PATH = "models/random_forest_model.joblib"
LABEL_ENCODER_PATH = "models/label_encoder_rf.pkl"
IMAGES_DIR = "data/NGT_Gestures"

SEQUENCE_LENGTH = 15     # Smaller buffer for RF stabilization
NUM_FEATURES = 63        # 21 landmarks * 3 coordinates
MIN_FRAMES = 5
PREDICTION_INTERVAL = 3
CONFIDENCE_THRESHOLD = 0.65

SIGNING_ZONE = {
    'x_min': -2.5,
    'x_max': 2.5,
    'y_min': -0.7,
    'y_max': 1.7,
}

ALL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# Page Config

st.set_page_config(
    page_title="NGT Sign Language",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS

st.markdown("""
<style>
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #a0a0b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .big-letter {
        font-size: 8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    
    .confidence-text {
        font-size: 1.5rem;
        color: #10b981;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Shared State

class SharedState:
    def __init__(self):
        self.stable_letter = None
        self.stable_confidence = 0.0

shared_state = SharedState()


# Helper Functions

@st.cache_resource
def load_model():
    """Load the trained Random Forest model and label encoder."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        return None, None
    
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    return model, label_encoder


@st.cache_resource
def init_mediapipe():
    """Initialize MediaPipe resources"""
    return detection.initialize_mediapipe()


def is_in_signing_zone(relative_position):
    """Check if hand is in signing zone"""
    if relative_position is None:
        return False
    
    x = relative_position.get('rel_x', 0)
    y = relative_position.get('rel_y', 0)
    
    return (SIGNING_ZONE['x_min'] <= x <= SIGNING_ZONE['x_max'] and
            SIGNING_ZONE['y_min'] <= y <= SIGNING_ZONE['y_max'])


def prepare_frame_features(landmarks_normalized):
    """Flatten landmarks into a single feature row"""
    features = []
    for lm in landmarks_normalized:
        features.extend([lm['x'], lm['y'], lm['z']])
    return np.array(features).reshape(1, -1)


def load_guide_image(letter):
    """Load reference image for a letter"""
    img_path = os.path.join(IMAGES_DIR, f"{letter}.jpg")
    if os.path.exists(img_path):
        return img_path
    return None


# Video Processor

class SignProcessor(VideoProcessorBase):
    """Real-time processor using a Random Forest model"""
    
    def __init__(self):
        self.sequence_data = deque(maxlen=SEQUENCE_LENGTH)
        self.frame_count = 0
        self.model, self.label_encoder = load_model()
        self.mp_resources = init_mediapipe()
        self.mirrored = True
    
    def recv(self, frame):
        """Process each video frame"""
        img = frame.to_ndarray(format="bgr24")
        
        if self.mirrored:
            img = cv2.flip(img, 1)
        
        annotated_frame, face_refs, hands_data = detection.process_frame(img, self.mp_resources)
        
        face_ok = face_refs is not None
        hand_ok = len(hands_data) > 0
        in_zone = False
        
        if hand_ok and face_ok:
            hand_data = hands_data[0]
            rel_pos = hand_data.get('relative_position')
            in_zone = is_in_signing_zone(rel_pos)
            
            if in_zone:
                # Store normalized landmarks
                frame_features = []
                for lm in hand_data['landmarks_normalized']:
                    frame_features.extend([lm['x'], lm['y'], lm['z']])
                self.sequence_data.append(frame_features)
                
                self.frame_count += 1
                
                # Random Forest Prediction
                if len(self.sequence_data) >= MIN_FRAMES and self.frame_count % PREDICTION_INTERVAL == 0:
                    
                    if self.model is not None:
                        # Use current frame for RF prediction
                        features = prepare_frame_features(hand_data['landmarks_normalized'])
                        probabilities = self.model.predict_proba(features)[0]
                        
                        predicted_idx = np.argmax(probabilities)
                        confidence = probabilities[predicted_idx]
                        
                        if confidence >= CONFIDENCE_THRESHOLD:
                            predicted_letter = self.label_encoder.inverse_transform([predicted_idx])[0]
                            shared_state.stable_letter = predicted_letter
                            shared_state.stable_confidence = confidence
            else:
                self.sequence_data.clear()
        else:
            self.sequence_data.clear()
        
        # Draw prediction on frame
        if shared_state.stable_letter:
            cv2.rectangle(annotated_frame, (10, 10), (220, 110), (30, 30, 60), -1)
            cv2.rectangle(annotated_frame, (10, 10), (220, 110), (99, 102, 241), 2)
            cv2.putText(annotated_frame, shared_state.stable_letter, (40, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (99, 102, 241), 4)
            cv2.putText(annotated_frame, f"{shared_state.stable_confidence:.0%}", (140, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (16, 185, 129), 2)
        
        # Status indicators
        y_offset = 120
        cv2.putText(annotated_frame, f"Face: {'OK' if face_ok else 'NO'}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (16, 185, 129) if face_ok else (239, 68, 68), 1)
        
        cv2.putText(annotated_frame, f"Hand: {'OK' if hand_ok else 'NO'}", 
                   (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (16, 185, 129) if hand_ok else (239, 68, 68), 1)
        
        cv2.putText(annotated_frame, f"Zone: {'OK' if in_zone else 'NO'}", 
                   (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (16, 185, 129) if in_zone else (245, 158, 11), 1)
        
        cv2.putText(annotated_frame, f"Buffer: {len(self.sequence_data)}/{SEQUENCE_LENGTH}", 
                   (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 176), 1)
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# Session State

if 'word' not in st.session_state:
    st.session_state.word = ""

if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "Practice"

if 'target_letter' not in st.session_state:
    st.session_state.target_letter = "A"


# Sidebar

with st.sidebar:
    st.markdown("### 🤟 NGT Sign Language")
    st.markdown("---")
    
    mode = st.radio(
        "Select Mode",
        ["Practice", "Learn"],
        index=0 if st.session_state.current_mode == "Practice" else 1
    )
    st.session_state.current_mode = mode
    
    st.markdown("---")
    
    st.markdown("### 📝 Word Builder")
    st.text_input("Current Word", value=st.session_state.word, key="word_display", disabled=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add Letter", use_container_width=True):
            if shared_state.stable_letter:
                st.session_state.word += shared_state.stable_letter
                st.rerun()
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.word = ""
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📊 Status")
    model, _ = load_model()
    model_status = "✅ Loaded" if model else "❌ Not Found"
    st.markdown(f"**Model:** {model_status}")
    
    if shared_state.stable_letter:
        st.markdown(f"**Last Detection:** {shared_state.stable_letter} ({shared_state.stable_confidence:.0%})")


# Main Content

st.markdown('<h1 class="main-title">🤟 NGT Sign Language</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Learn Dutch fingerspelling with AI-powered feedback</p>', unsafe_allow_html=True)

model, label_encoder = load_model()

if model is None:
    st.error("⚠️ Model not found! Please train the unified model first.")
    st.stop()


# Practice Mode

if st.session_state.current_mode == "Practice":
    st.markdown("### 📹 Live Camera Feed")
    st.info("👋 Click **START** to begin. Use the sidebar to switch modes.")
    
    webrtc_ctx = webrtc_streamer(
        key="ngt-sign-main",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=SignProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_html_attrs={
            "style": {"width": "100%", "margin-top": "10px"},
            "controls": False,
            "autoPlay": True,
        },
    )

    if webrtc_ctx and webrtc_ctx.state.playing:
        st.markdown("---")
        pred_placeholder = st.empty()
        
        if shared_state.stable_letter:
            confidence_pct = int(shared_state.stable_confidence * 100)
            color = "green" if confidence_pct > 60 else "orange"
            pred_placeholder.markdown(
                f"<h1 style='text-align: center; font-size: 3rem;'>Predicted: <span style='color:{color}'>{shared_state.stable_letter}</span> <span style='font-size: 1.5rem'>({confidence_pct}%)</span></h1>", 
                unsafe_allow_html=True
            )
        else:
            pred_placeholder.markdown(
                "<h3 style='text-align: center; color: gray;'>Waiting for sign...</h3>", 
                unsafe_allow_html=True
            )


# Learn Mode

elif st.session_state.current_mode == "Learn":
    st.markdown("### Select a Letter to Learn")
    cols = st.columns(13)
    for i, letter in enumerate(ALL_LETTERS):
        with cols[i % 13]:
            if st.button(letter, key=f"learn_{letter}", use_container_width=True,
                        type="primary" if letter == st.session_state.target_letter else "secondary"):
                st.session_state.target_letter = letter
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📚 How to Sign: {st.session_state.target_letter}")
        
        img_path = load_guide_image(st.session_state.target_letter)
        if img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"Reference image not found: {st.session_state.target_letter}.jpg")
            st.info(f"Please add image to {IMAGES_DIR}/")
    
    with col2:
        st.markdown("### 📹 Your Practice")
        st.info("👆 Click **START** to begin the live video feed")
        
        webrtc_ctx = webrtc_streamer(
            key="ngt-sign-main",  # Using the same key to avoid camera conflicts
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_processor_factory=SignProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            video_html_attrs={
                "style": {"width": "100%", "margin-top": "10px"},
                "controls": False,
                "autoPlay": True,
            },
        )
        
        if shared_state.stable_letter:
            if shared_state.stable_letter == st.session_state.target_letter:
                st.success(f"✅ Perfect! You signed '{st.session_state.target_letter}' correctly!")
            else:
                st.warning(f"🤔 Detected '{shared_state.stable_letter}' - Keep practicing '{st.session_state.target_letter}'")


# Footer

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>NGT Sign Language Recognition | "
    "Part of the ADS&AI Sign Language Challenge at Breda University of Applied Sciences</p>",
    unsafe_allow_html=True
)