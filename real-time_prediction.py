"""
NGT Sign Language Recognition - Real-Time Prediction
Recognizes static hand gestures using trained Random Forest model.

Features:
    - Real-time hand landmark detection
    - Letter prediction with confidence display
    - Stabilized predictions (reduces flickering)
    - Signing zone validation
    - Display on webcam window + console output

Controls:
    - Press 'q' to quit
    - Press 'm' to toggle mirror mode
    - Press 'z' to toggle signing zone visibility

Usage:
    python realtime_prediction.py
"""

import cv2
import numpy as np
import joblib
import os
from collections import deque

import hand_face_detection as detection


# =============================================================================
# CONFIGURATION (Adjust these parameters as needed)
# =============================================================================

# Model path
MODEL_PATH = "models/random_forest.joblib"

# Prediction stabilization
# Only display prediction when same letter detected N times in a row
STABILITY_THRESHOLD = 10

# Confidence threshold
# Only display prediction if confidence >= this value (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.60

# Signing zone boundaries (relative to face)
# Hand must be within this zone for predictions to be made
SIGNING_ZONE = {
    'x_min': -2.5,   # Left boundary (face widths from nose)
    'x_max': 2.5,    # Right boundary
    'y_min': -0.7,   # Upper boundary (above nose)
    'y_max': 1.7,    # Lower boundary (below nose)
}

# Display settings
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

def load_model(model_path):
    """Load trained model from file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run train_model.ipynb first to train and save the model."
        )
    
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


# =============================================================================
# PREDICTION
# =============================================================================

def landmarks_to_features(landmarks_normalized):
    """
    Convert normalized landmarks to feature array for model input.
    
    Args:
        landmarks_normalized: List of 21 dicts with 'x', 'y' keys
    
    Returns:
        numpy array of shape (1, 42)
    """
    features = []
    for lm in landmarks_normalized:
        features.append(lm['x'])
        features.append(lm['y'])
    return np.array(features).reshape(1, -1)


def predict_letter(model, landmarks_normalized):
    """
    Predict letter from landmarks.
    
    Returns:
        (predicted_letter, confidence) or (None, 0.0) if prediction fails
    """
    try:
        features = landmarks_to_features(landmarks_normalized)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities.max()
        return prediction, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0


# =============================================================================
# PREDICTION STABILIZER
# =============================================================================

class PredictionStabilizer:
    """
    Stabilizes predictions by requiring N consecutive same predictions.
    Reduces flickering between letters.
    """
    
    def __init__(self, threshold=10):
        self.threshold = threshold
        self.history = deque(maxlen=threshold)
        self.stable_prediction = None
        self.stable_confidence = 0.0
    
    def update(self, prediction, confidence):
        """
        Update with new prediction.
        
        Returns:
            (stable_letter, stable_confidence, is_stable)
        """
        self.history.append((prediction, confidence))
        
        # Check if all recent predictions are the same
        if len(self.history) == self.threshold:
            predictions = [p[0] for p in self.history]
            
            if all(p == predictions[0] and p is not None for p in predictions):
                self.stable_prediction = predictions[0]
                self.stable_confidence = np.mean([p[1] for p in self.history])
                return self.stable_prediction, self.stable_confidence, True
        
        return self.stable_prediction, self.stable_confidence, False
    
    def reset(self):
        """Reset the stabilizer."""
        self.history.clear()
        self.stable_prediction = None
        self.stable_confidence = 0.0


# =============================================================================
# SIGNING ZONE CHECK
# =============================================================================

def is_in_signing_zone(relative_position):
    """
    Check if hand is within the signing zone.
    
    Args:
        relative_position: Dict with 'rel_x', 'rel_y' keys
    
    Returns:
        bool
    """
    if relative_position is None:
        return False
    
    x = relative_position['rel_x']
    y = relative_position['rel_y']
    
    return (SIGNING_ZONE['x_min'] <= x <= SIGNING_ZONE['x_max'] and
            SIGNING_ZONE['y_min'] <= y <= SIGNING_ZONE['y_max'])


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def draw_signing_zone(frame, face_refs, show_zone=True):
    """Draw signing zone rectangle based on face position."""
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


def draw_prediction(frame, letter, confidence, is_stable, in_zone):
    """Draw prediction result on frame."""
    h, w, _ = frame.shape
    
    # Main prediction display area (top-right corner)
    box_x = w - 200
    box_y = 10
    box_w = 190
    box_h = 120
    
    # Background
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_WHITE, 2)
    
    if not in_zone:
        # Hand not in zone
        cv2.putText(frame, "MOVE HAND", (box_x + 20, box_y + 50),
                   DISPLAY_FONT, 0.7, COLOR_ORANGE, 2)
        cv2.putText(frame, "INTO ZONE", (box_x + 25, box_y + 80),
                   DISPLAY_FONT, 0.7, COLOR_ORANGE, 2)
    elif letter is None:
        # No stable prediction yet
        cv2.putText(frame, "DETECTING", (box_x + 20, box_y + 50),
                   DISPLAY_FONT, 0.7, COLOR_YELLOW, 2)
        cv2.putText(frame, "...", (box_x + 80, box_y + 80),
                   DISPLAY_FONT, 0.7, COLOR_YELLOW, 2)
    elif confidence < CONFIDENCE_THRESHOLD:
        # Low confidence
        cv2.putText(frame, f"{letter}?", (box_x + 60, box_y + 70),
                   DISPLAY_FONT, 2.0, COLOR_ORANGE, 3)
        cv2.putText(frame, f"{confidence:.0%} LOW", (box_x + 40, box_y + 100),
                   DISPLAY_FONT, 0.5, COLOR_ORANGE, 1)
    else:
        # Good prediction
        color = COLOR_GREEN if is_stable else COLOR_CYAN
        cv2.putText(frame, letter, (box_x + 60, box_y + 75),
                   DISPLAY_FONT, 2.5, color, 4)
        cv2.putText(frame, f"{confidence:.0%}", (box_x + 70, box_y + 105),
                   DISPLAY_FONT, 0.6, color, 2)
    
    return frame


def draw_status(frame, face_detected, hand_detected, in_zone, mirrored, show_zone):
    """Draw status bar at bottom of frame."""
    h, w, _ = frame.shape
    
    # Status bar background
    cv2.rectangle(frame, (10, h - 40), (500, h - 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, h - 40), (500, h - 10), COLOR_WHITE, 1)
    
    # Status indicators
    face_color = COLOR_GREEN if face_detected else COLOR_RED
    hand_color = COLOR_GREEN if hand_detected else COLOR_RED
    zone_color = COLOR_GREEN if in_zone else COLOR_RED
    
    status_text = (
        f"Face: {'OK' if face_detected else 'NO'} | "
        f"Hand: {'OK' if hand_detected else 'NO'} | "
        f"Zone: {'OK' if in_zone else 'NO'} | "
        f"Mirror: {'ON' if mirrored else 'OFF'} (m) | "
        f"Zone: {'ON' if show_zone else 'OFF'} (z)"
    )
    
    cv2.putText(frame, status_text, (20, h - 18),
               DISPLAY_FONT, 0.4, COLOR_WHITE, 1)
    
    return frame


def draw_controls(frame):
    """Draw control hints."""
    h, w, _ = frame.shape
    
    cv2.putText(frame, "Press 'q' to quit", (10, 25),
               DISPLAY_FONT, 0.5, COLOR_WHITE, 1)
    
    return frame


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main real-time prediction loop."""
    
    print("=" * 60)
    print("NGT Sign Language Recognition - Real-Time Prediction")
    print("=" * 60)
    print(f"Stability threshold: {STABILITY_THRESHOLD} frames")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = load_model(MODEL_PATH)
    
    # Initialize MediaPipe
    print("Initializing detection...")
    mp_resources = detection.initialize_mediapipe()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize stabilizer
    stabilizer = PredictionStabilizer(threshold=STABILITY_THRESHOLD)
    
    # State
    mirrored = True
    show_zone = True
    last_printed_letter = None
    
    print("\nReady! Controls:")
    print("  q - Quit")
    print("  m - Toggle mirror mode")
    print("  z - Toggle signing zone display")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if mirrored:
            frame = cv2.flip(frame, 1)
        
        # Detect face and hands
        frame, face_refs, hands_data = detection.process_frame(frame, mp_resources)
        
        # State tracking
        face_detected = face_refs is not None
        hand_detected = len(hands_data) > 0
        in_zone = False
        current_prediction = None
        current_confidence = 0.0
        
        # Process hand if detected
        if hand_detected and face_detected:
            hand_data = hands_data[0]  # Use first detected hand
            rel_pos = hand_data.get('relative_position')
            
            in_zone = is_in_signing_zone(rel_pos)
            
            if in_zone:
                # Get prediction
                landmarks_norm = hand_data['landmarks_normalized']
                current_prediction, current_confidence = predict_letter(model, landmarks_norm)
        
        # Update stabilizer
        if in_zone and current_prediction is not None:
            stable_letter, stable_confidence, is_stable = stabilizer.update(
                current_prediction, current_confidence
            )
        else:
            stabilizer.reset()
            stable_letter, stable_confidence, is_stable = None, 0.0, False
        
        # Print to console (only when stable prediction changes)
        if (stable_letter is not None and 
            stable_confidence >= CONFIDENCE_THRESHOLD and
            stable_letter != last_printed_letter):
            print(f"Detected: {stable_letter} ({stable_confidence:.0%})")
            last_printed_letter = stable_letter
        
        # Reset last printed if hand leaves zone
        if not in_zone:
            last_printed_letter = None
        
        # Draw UI elements
        frame = draw_signing_zone(frame, face_refs, show_zone)
        frame = draw_prediction(frame, stable_letter, stable_confidence, is_stable, in_zone)
        frame = draw_status(frame, face_detected, hand_detected, in_zone, mirrored, show_zone)
        frame = draw_controls(frame)
        
        # Draw hand landmarks
        if hands_data:
            for hand_data in hands_data:
                center = hand_data['center_px']
                color = COLOR_GREEN if in_zone else COLOR_RED
                cv2.circle(frame, center, 10, color, -1)
        
        # Show frame
        cv2.imshow('NGT Sign Language Recognition', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mirrored = not mirrored
            print(f"Mirror mode: {'ON' if mirrored else 'OFF'}")
        elif key == ord('z'):
            show_zone = not show_zone
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    mp_resources['hands'].close()
    mp_resources['face_mesh'].close()
    print("\nStopped.")

# Make file import-friendly
if __name__ == "__main__":
    main()