# Test real-time fingerspelling

import cv2
import numpy as np
import joblib
import os
from collections import deque
import hand_face_detection as detection


# Configuration

MODEL_PATH = "models/random_forest_model.joblib"
LABEL_ENCODER_PATH = "models/label_encoder_rf.pkl"

SEQUENCE_LENGTH = 15  # Smaller buffer for RF stabilization
MIN_FRAMES = 5
PREDICTION_INTERVAL = 3

CONFIDENCE_THRESHOLD = 0.65

SIGNING_ZONE = {
    'x_min': -2.5,
    'x_max': 2.5,
    'y_min': -0.7,
    'y_max': 1.7,
}

DISPLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)


# Load model

def load_model(model_path, label_encoder_path):
    """Load trained Random Forest model"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")
    
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    
    print(f"Random Forest model loaded: {model_path}")
    print(f"Supports {len(label_encoder.classes_)} letters: {list(label_encoder.classes_)}")
    
    return model, label_encoder


# Feature Extraction

def prepare_frame_features(landmarks_normalized, landmarks_absolute):
    """Flatten landmarks into a single feature row (63 + 3 = 66)"""
    features = []
    # 1. Add normalized landmarks (relative to wrist)
    for lm in landmarks_normalized:
        features.extend([lm['x'], lm['y'], lm['z']])
    
    # 2. Add absolute wrist position (landmark 0 before normalization)
    wrist_abs = landmarks_absolute[0]
    features.extend([wrist_abs['x'], wrist_abs['y'], wrist_abs['z']])
    
    return np.array(features).reshape(1, -1)


# Sequence Buffer with Wrist Tracking

class UnifiedSequenceBuffer:
    """Buffer that tracks both normalized landmarks and absolute wrist positions"""
    
    def __init__(self, max_length=75):
        self.max_length = max_length
        self.frames_norm = deque(maxlen=max_length)
        self.frames_wrist_abs = deque(maxlen=max_length)
    
    def add_frame(self, landmarks_normalized, landmarks_absolute):
        """Add frame with both normalized and absolute data"""
        # Normalized landmarks
        frame_norm = []
        for lm in landmarks_normalized:
            frame_norm.extend([lm['x'], lm['y'], lm['z']])
        self.frames_norm.append(frame_norm)
        
        # Absolute wrist position (landmark 0 before normalization)
        wrist_abs = landmarks_absolute[0]
        self.frames_wrist_abs.append([wrist_abs['x'], wrist_abs['y'], wrist_abs['z']])
    
    def get_sequences(self):
        """Get both sequences as numpy arrays"""
        if len(self.frames_norm) == 0:
            return None, None
        
        seq_norm = np.array(list(self.frames_norm), dtype=np.float32)
        seq_wrist = np.array(list(self.frames_wrist_abs), dtype=np.float32)
        
        return seq_norm, seq_wrist
    
    def clear(self):
        """Clear both buffers"""
        self.frames_norm.clear()
        self.frames_wrist_abs.clear()
    
    def is_ready(self, min_frames=30):
        """Check if enough frames collected"""
        return len(self.frames_norm) >= min_frames
    
    def __len__(self):
        return len(self.frames_norm)


# Prediction

def predict(model, landmarks_normalized, landmarks_absolute, label_encoder):
    """
    Predict letter using Random Forest
    """
    
    features = prepare_frame_features(landmarks_normalized, landmarks_absolute)
    
    # Get probability across all classes
    probabilities = model.predict_proba(features)[0]
    
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx]
    predicted_letter = label_encoder.inverse_transform([predicted_idx])[0]
    
    return predicted_letter, confidence


# Signing Zone

def is_in_signing_zone(relative_position):
    """Check if hand is in signing zone"""
    if relative_position is None:
        return False
    
    x = relative_position['rel_x']
    y = relative_position['rel_y']
    
    return (SIGNING_ZONE['x_min'] <= x <= SIGNING_ZONE['x_max'] and
            SIGNING_ZONE['y_min'] <= y <= SIGNING_ZONE['y_max'])


def draw_signing_zone(frame, face_refs, show_zone=True):
    """Draw signing zone rectangle"""
    if not show_zone or face_refs is None:
        return frame
    
    nose = face_refs['nose']
    fw = face_refs['face_width']
    fh = face_refs['face_height']
    
    x1 = int(nose[0] + SIGNING_ZONE['x_min'] * fw)
    x2 = int(nose[0] + SIGNING_ZONE['x_max'] * fw)
    y1 = int(nose[1] + SIGNING_ZONE['y_min'] * fh)
    y2 = int(nose[1] + SIGNING_ZONE['y_max'] * fh)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_YELLOW, 2)
    return frame


# Display

def draw_prediction_box(frame, letter, confidence, buffer_size, in_zone):
    """Draw prediction result box"""
    h, w, _ = frame.shape
    
    box_x, box_y = w - 220, 10
    box_w, box_h = 210, 140
    
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_WHITE, 2)
    
    # Buffer fill bar
    fill_pct = min(buffer_size / SEQUENCE_LENGTH, 1.0)
    bar_width = int((box_w - 20) * fill_pct)
    cv2.rectangle(frame, (box_x + 10, box_y + box_h - 20),
                 (box_x + 10 + bar_width, box_y + box_h - 10), COLOR_CYAN, -1)
    cv2.rectangle(frame, (box_x + 10, box_y + box_h - 20),
                 (box_x + box_w - 10, box_y + box_h - 10), COLOR_WHITE, 1)
    
    if not in_zone:
        cv2.putText(frame, "MOVE HAND", (box_x + 35, box_y + 50),
                   DISPLAY_FONT, 0.6, COLOR_RED, 2)
        cv2.putText(frame, "INTO ZONE", (box_x + 40, box_y + 75),
                   DISPLAY_FONT, 0.6, COLOR_RED, 2)
    elif letter is None:
        cv2.putText(frame, "Detecting...", (box_x + 40, box_y + 60),
                   DISPLAY_FONT, 0.6, COLOR_YELLOW, 2)
    elif confidence < CONFIDENCE_THRESHOLD:
        cv2.putText(frame, f"{letter}?", (box_x + 70, box_y + 65),
                   DISPLAY_FONT, 1.8, COLOR_ORANGE, 3)
        cv2.putText(frame, f"{confidence:.0%}", (box_x + 80, box_y + 95),
                   DISPLAY_FONT, 0.5, COLOR_ORANGE, 1)
    else:
        cv2.putText(frame, letter, (box_x + 70, box_y + 70),
                   DISPLAY_FONT, 2.2, COLOR_GREEN, 4)
        cv2.putText(frame, f"{confidence:.0%}", (box_x + 80, box_y + 100),
                   DISPLAY_FONT, 0.6, COLOR_GREEN, 2)
    
    return frame


def draw_status_bar(frame, face_ok, hand_ok, in_zone, mirrored, fps):
    """Draw status bar at bottom"""
    h, w, _ = frame.shape
    
    cv2.rectangle(frame, (10, h - 35), (500, h - 10), (0, 0, 0), -1)
    
    status = f"Face:{'OK' if face_ok else 'NO'} Hand:{'OK' if hand_ok else 'NO'} "
    status += f"Zone:{'OK' if in_zone else 'NO'} Mirror:{'ON' if mirrored else 'OFF'} FPS:{fps:.0f}"
    
    cv2.putText(frame, status, (15, h - 17), DISPLAY_FONT, 0.4, COLOR_WHITE, 1)
    cv2.putText(frame, "Q=quit M=mirror Z=zone | RF: All 26 Letters", (280, h - 17), 
               DISPLAY_FONT, 0.35, (150, 150, 150), 1)
    
    return frame

# Main

def main():
    
    model, label_encoder = load_model(MODEL_PATH, LABEL_ENCODER_PATH)
    
    print("\nInitializing MediaPipe...")
    mp_resources = detection.initialize_mediapipe()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    sequence_buffer = UnifiedSequenceBuffer(max_length=SEQUENCE_LENGTH)
    
    mirrored = True
    show_zone = True
    frame_count = 0
    last_printed_letter = None
    
    current_letter = None
    current_confidence = 0.0
    
    fps = 0
    fps_counter = 0
    fps_start = cv2.getTickCount()
    
    print("\nReady! Random Forest model recognizes all 26 letters")
    print("Controls: Q=quit, M=mirror, Z=zone")
    print("Supports both static and dynamic gestures\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if mirrored:
            frame = cv2.flip(frame, 1)
        
        frame, face_refs, hands_data = detection.process_frame(frame, mp_resources)
        
        face_ok = face_refs is not None
        hand_ok = len(hands_data) > 0
        in_zone = False
        
        if hand_ok and face_ok:
            hand_data = hands_data[0]
            rel_pos = hand_data.get('relative_position')
            in_zone = is_in_signing_zone(rel_pos)
            
            if in_zone:
                # Add frame with both normalized and absolute landmarks
                sequence_buffer.add_frame(
                    hand_data['landmarks_normalized'],
                    hand_data['landmarks']
                )
                
                # Predict every N frames when buffer ready
                if sequence_buffer.is_ready(MIN_FRAMES) and frame_count % PREDICTION_INTERVAL == 0:
                    
                    predicted_letter, confidence = predict(
                        model, 
                        hand_data['landmarks_normalized'], 
                        hand_data['landmarks'],
                        label_encoder
                    )
                    
                    current_letter = predicted_letter
                    current_confidence = confidence
                        
                    if (predicted_letter != last_printed_letter and confidence >= CONFIDENCE_THRESHOLD):
                            print(f"Detected: {predicted_letter} ({confidence:.0%})")
                            last_printed_letter = predicted_letter
            else:
                sequence_buffer.clear()
                current_letter = None
                current_confidence = 0.0
                last_printed_letter = None
        else:
            sequence_buffer.clear()
            current_letter = None
            current_confidence = 0.0
            last_printed_letter = None
        
        frame = draw_signing_zone(frame, face_refs, show_zone)
        frame = draw_prediction_box(frame, current_letter, current_confidence,
                                    len(sequence_buffer), in_zone)
        frame = draw_status_bar(frame, face_ok, hand_ok, in_zone, mirrored, fps)
        
        if hand_ok:
            center = hands_data[0]['center_px']
            color = COLOR_GREEN if in_zone else COLOR_RED
            cv2.circle(frame, center, 8, color, -1)
        
        cv2.imshow('NGT Sign Language Recognition - Random Forest', frame)
        
        frame_count += 1
        fps_counter += 1
        if fps_counter >= 30:
            fps_end = cv2.getTickCount()
            fps = 30 / ((fps_end - fps_start) / cv2.getTickFrequency())
            fps_start = fps_end
            fps_counter = 0
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mirrored = not mirrored
            print(f"Mirror: {'ON' if mirrored else 'OFF'}")
        elif key == ord('z'):
            show_zone = not show_zone
    
    cap.release()
    cv2.destroyAllWindows()
    mp_resources['hands'].close()
    mp_resources['face_mesh'].close()
    print("\nStopped.")


if __name__ == "__main__":
    main()