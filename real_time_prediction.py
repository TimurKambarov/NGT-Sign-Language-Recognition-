"""
NGT Sign Language Recognition - Real-Time LightGBM Inference
Uses LightGBM model with feature engineering for real-time webcam prediction

Controls:
    Q - Quit
    M - Toggle mirror mode
    Z - Toggle signing zone visibility

Usage:
    python real_time_prediction_lightgbm.py
"""

import cv2
import numpy as np
import lightgbm as lgb
import joblib
import os
from collections import deque

try:
    import hand_face_detection as detection
except ImportError:
    print("Error: hand_face_detection.py not found in current directory.")
    print("Make sure you're running from the repository root.")
    exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model paths
MODEL_PATH = "models/lightgbm_model.txt"
LABEL_ENCODER_PATH = "models/label_encoder_lgbm.pkl"
FEATURE_INFO_PATH = "models/feature_info.pkl"

# Sequence settings
SEQUENCE_LENGTH = 10      # Collect 20 frames before prediction
PREDICTION_INTERVAL = 5   # Predict every 5 frames

# Prediction settings
CONFIDENCE_THRESHOLD = 0.70

# Signing zone boundaries (relative to face)
SIGNING_ZONE = {
    'x_min': -2.5,
    'x_max': 2.5,
    'y_min': -0.7,
    'y_max': 1.7,
}

# Display
DISPLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path, label_encoder_path):
    """Load trained LightGBM model and label encoder"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train the model first using the Jupyter notebook."
        )
    
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")
    
    # Load LightGBM model (native format)
    model = lgb.Booster(model_file=model_path)
    
    # Load label encoder
    label_encoder = joblib.load(label_encoder_path)
    
    print(f"✓ Model loaded: {model_path}")
    print(f"✓ Classes: {list(label_encoder.classes_)}")
    
    return model, label_encoder


# =============================================================================
# FEATURE EXTRACTION (must match training!)
# =============================================================================

def extract_features(sequence):
    """
    Extract statistical features from sequence
    
    For each of 60 coordinates (excluding x0, y0, z0):
    - Mean: Average value
    - Std: Standard deviation
    - Min: Minimum value
    - Max: Maximum value
    
    Returns: 1D array of 240 features
    
    CHANGED: Skip first 3 columns (x0, y0, z0) - wrist is always (0, 0, 0)
    """
    features = []
    
    # Start from index 3 to skip wrist coordinates (always 0)
    for coord_idx in range(3, sequence.shape[1]):  # Changed: start from 3 instead of 0
        coord_values = sequence[:, coord_idx]
        
        features.append(np.mean(coord_values))
        features.append(np.std(coord_values))
        features.append(np.min(coord_values))
        features.append(np.max(coord_values))
    
    return np.array(features).reshape(1, -1)  # Shape: (1, 240)


# =============================================================================
# SEQUENCE BUFFER
# =============================================================================

class SequenceBuffer:
    """Maintains buffer of recent frames"""
    
    def __init__(self, max_length=20):
        self.max_length = max_length
        self.frames = deque(maxlen=max_length)
    
    def add_frame(self, landmarks_normalized):
        """Add frame (21 landmarks with x, y, z)"""
        frame_data = []
        for lm in landmarks_normalized:
            frame_data.extend([lm['x'], lm['y'], lm['z']])
        self.frames.append(frame_data)
    
    def get_sequence(self):
        """Get sequence as numpy array (num_frames, 63)"""
        if len(self.frames) == 0:
            return None
        return np.array(list(self.frames), dtype=np.float32)
    
    def clear(self):
        """Clear buffer"""
        self.frames.clear()
    
    def is_ready(self):
        """Check if buffer is full"""
        return len(self.frames) >= self.max_length
    
    def __len__(self):
        return len(self.frames)


# =============================================================================
# PREDICTION
# =============================================================================

def predict(model, sequence, label_encoder):
    """
    Predict letter from sequence using LightGBM
    Returns: (predicted_letter, confidence)
    """
    
    # Extract features from sequence
    features = extract_features(sequence)
    
    # Predict with LightGBM
    predictions = model.predict(features)[0]  # Returns probabilities
    
    # Get best prediction
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx]
    predicted_letter = label_encoder.inverse_transform([predicted_idx])[0]
    
    return predicted_letter, confidence


# =============================================================================
# SIGNING ZONE
# =============================================================================

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


# =============================================================================
# DISPLAY
# =============================================================================

def draw_prediction_box(frame, letter, confidence, buffer_size, in_zone):
    """Draw prediction result box"""
    h, w, _ = frame.shape
    
    box_x, box_y = w - 200, 10
    box_w, box_h = 190, 130
    
    # Background
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_WHITE, 2)
    
    # Buffer fill bar
    fill_pct = min(buffer_size / SEQUENCE_LENGTH, 1.0)
    bar_width = int((box_w - 20) * fill_pct)
    cv2.rectangle(frame, (box_x + 10, box_y + box_h - 20),
                 (box_x + 10 + bar_width, box_y + box_h - 10), COLOR_CYAN, -1)
    cv2.rectangle(frame, (box_x + 10, box_y + box_h - 20),
                 (box_x + box_w - 10, box_y + box_h - 10), COLOR_WHITE, 1)
    
    # Display prediction or status
    if not in_zone:
        cv2.putText(frame, "MOVE HAND", (box_x + 25, box_y + 45),
                   DISPLAY_FONT, 0.6, COLOR_RED, 2)
        cv2.putText(frame, "INTO ZONE", (box_x + 30, box_y + 70),
                   DISPLAY_FONT, 0.6, COLOR_RED, 2)
    elif letter is None:
        cv2.putText(frame, "Detecting...", (box_x + 30, box_y + 55),
                   DISPLAY_FONT, 0.6, COLOR_YELLOW, 2)
    elif confidence < CONFIDENCE_THRESHOLD:
        # Low confidence
        cv2.putText(frame, f"{letter}?", (box_x + 60, box_y + 60),
                   DISPLAY_FONT, 1.8, COLOR_ORANGE, 3)
        cv2.putText(frame, f"{confidence:.0%}", (box_x + 70, box_y + 90),
                   DISPLAY_FONT, 0.5, COLOR_ORANGE, 1)
    else:
        # High confidence
        cv2.putText(frame, letter, (box_x + 60, box_y + 65),
                   DISPLAY_FONT, 2.2, COLOR_GREEN, 4)
        cv2.putText(frame, f"{confidence:.0%}", (box_x + 70, box_y + 95),
                   DISPLAY_FONT, 0.6, COLOR_GREEN, 2)
    
    return frame


def draw_status_bar(frame, face_ok, hand_ok, in_zone, mirrored, fps):
    """Draw status bar at bottom"""
    h, w, _ = frame.shape
    
    cv2.rectangle(frame, (10, h - 35), (500, h - 10), (0, 0, 0), -1)
    
    status = f"Face:{'OK' if face_ok else 'NO'} Hand:{'OK' if hand_ok else 'NO'} "
    status += f"Zone:{'OK' if in_zone else 'NO'} Mirror:{'ON' if mirrored else 'OFF'} FPS:{fps:.0f}"
    
    cv2.putText(frame, status, (15, h - 17), DISPLAY_FONT, 0.4, COLOR_WHITE, 1)
    cv2.putText(frame, "Q=quit M=mirror Z=zone", (350, h - 17), DISPLAY_FONT, 0.35, (150, 150, 150), 1)
    
    return frame


# =============================================================================
# MAIN
# =============================================================================

def main():
    
    # Load model and label encoder
    model, label_encoder = load_model(MODEL_PATH, LABEL_ENCODER_PATH)
    
    # Initialize detection
    print("\nInitializing MediaPipe...")
    mp_resources = detection.initialize_mediapipe()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # State
    sequence_buffer = SequenceBuffer(max_length=SEQUENCE_LENGTH)
    
    mirrored = True
    show_zone = True
    frame_count = 0
    last_printed_letter = None
    
    # Current prediction
    current_letter = None
    current_confidence = 0.0
    
    # FPS tracking
    fps = 0
    fps_counter = 0
    fps_start = cv2.getTickCount()
    
    print("\n✓ Ready! Controls: Q=quit, M=mirror, Z=zone")
    print("Predictions appear when hand is in zone.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if mirrored:
            frame = cv2.flip(frame, 1)
        
        # Detect face and hands
        frame, face_refs, hands_data = detection.process_frame(frame, mp_resources)
        
        face_ok = face_refs is not None
        hand_ok = len(hands_data) > 0
        in_zone = False
        
        # Process hand if detected
        if hand_ok and face_ok:
            hand_data = hands_data[0]
            rel_pos = hand_data.get('relative_position')
            in_zone = is_in_signing_zone(rel_pos)
            
            if in_zone:
                # Add frame to buffer
                sequence_buffer.add_frame(hand_data['landmarks_normalized'])
                
                # Predict every 5 frames when buffer is ready
                if sequence_buffer.is_ready() and frame_count % PREDICTION_INTERVAL == 0:
                    sequence = sequence_buffer.get_sequence()
                    
                    # Make prediction
                    predicted_letter, confidence = predict(
                        model, sequence, label_encoder
                    )
                    
                    current_letter = predicted_letter
                    current_confidence = confidence
                    
                    # Print to console on change (only high confidence)
                    if (predicted_letter != last_printed_letter and 
                        confidence >= CONFIDENCE_THRESHOLD):
                        print(f"Detected: {predicted_letter} ({confidence:.0%})")
                        last_printed_letter = predicted_letter
            else:
                # Hand left zone - clear everything
                sequence_buffer.clear()
                current_letter = None
                current_confidence = 0.0
                last_printed_letter = None
        else:
            # No detection - clear everything
            sequence_buffer.clear()
            current_letter = None
            current_confidence = 0.0
            last_printed_letter = None
        
        # Draw UI
        frame = draw_signing_zone(frame, face_refs, show_zone)
        frame = draw_prediction_box(frame, current_letter, current_confidence,
                                    len(sequence_buffer), in_zone)
        frame = draw_status_bar(frame, face_ok, hand_ok, in_zone, mirrored, fps)
        
        # Hand indicator
        if hand_ok:
            center = hands_data[0]['center_px']
            color = COLOR_GREEN if in_zone else COLOR_RED
            cv2.circle(frame, center, 8, color, -1)
        
        cv2.imshow('NGT Sign Language Recognition - LightGBM', frame)
        
        # FPS calculation
        frame_count += 1
        fps_counter += 1
        if fps_counter >= 30:
            fps_end = cv2.getTickCount()
            fps = 30 / ((fps_end - fps_start) / cv2.getTickFrequency())
            fps_start = fps_end
            fps_counter = 0
        
        # Input handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mirrored = not mirrored
            print(f"Mirror: {'ON' if mirrored else 'OFF'}")
        elif key == ord('z'):
            show_zone = not show_zone
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    mp_resources['hands'].close()
    mp_resources['face_mesh'].close()
    print("\n✓ Stopped.")


if __name__ == "__main__":
    main()