"""
NGT Sign Language Recognition - Data Collection Module
Collects hand landmark data for training the classifier.

Usage:
    1. Run the script
    2. Select a letter using number keys (shown on screen)
    3. Position your hand in the signing zone
    4. Press SPACEBAR to save a sample
    5. Press 'q' to quit and save all data

Output:
    - data/samples.csv: All collected samples with 42 features (21 landmarks × 2 coords) + label
"""

import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import hand_face_detection as detection


# =============================================================================
# CONFIGURATION
# =============================================================================

# Output settings
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "samples.csv")

# Static letters only (excluding dynamic: H, J, Z)
STATIC_LETTERS = list("ABCDEFGIKLMNOPQRSTUVWXY")

# Letters per row in the UI display
LETTERS_PER_ROW = 8

# Colors
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)


# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def create_column_names():
    """Create column names for the CSV: x0, y0, x1, y1, ..., x20, y20, label."""
    columns = []
    for i in range(21):
        columns.append(f"x{i}")
        columns.append(f"y{i}")
    columns.append("label")
    return columns


def landmarks_to_row(landmarks_normalized, label):
    """
    Convert normalized landmarks to a flat row for CSV.
    
    Args:
        landmarks_normalized: List of 21 dicts with 'x', 'y' keys (wrist-relative)
        label: Letter label (e.g., 'A')
    
    Returns:
        List of 43 values: [x0, y0, x1, y1, ..., x20, y20, label]
    """
    row = []
    for lm in landmarks_normalized:
        row.append(lm['x'])
        row.append(lm['y'])
    row.append(label)
    return row


def load_existing_data(filepath):
    """Load existing CSV data or create empty DataFrame."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} existing samples from {filepath}")
        return df
    else:
        columns = create_column_names()
        print("Starting with empty dataset")
        return pd.DataFrame(columns=columns)


def save_data(df, filepath):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} samples to {filepath}")


def get_sample_counts(df):
    """Get count of samples per letter."""
    if len(df) == 0:
        return {}
    return df['label'].value_counts().to_dict()


# =============================================================================
# UI DRAWING
# =============================================================================

def draw_letter_selector(frame, current_letter, sample_counts):
    """Draw the letter selection UI on the frame."""
    h, w, _ = frame.shape
    
    # Background panel
    panel_height = 140
    cv2.rectangle(frame, (10, h - panel_height - 10), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, h - panel_height - 10), (w - 10, h - 10), COLOR_WHITE, 2)
    
    # Title
    cv2.putText(frame, "Select letter (press number) | SPACE = save sample | Q = quit & save",
               (20, h - panel_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    
    # Draw letters in grid
    start_y = h - panel_height + 40
    box_size = 35
    margin = 5
    
    for idx, letter in enumerate(STATIC_LETTERS):
        row = idx // LETTERS_PER_ROW
        col = idx % LETTERS_PER_ROW
        
        x = 20 + col * (box_size + margin)
        y = start_y + row * (box_size + margin)
        
        # Highlight selected letter
        if letter == current_letter:
            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), COLOR_GREEN, -1)
            text_color = (0, 0, 0)
        else:
            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), COLOR_WHITE, 1)
            text_color = COLOR_WHITE
        
        # Letter and index
        cv2.putText(frame, letter, (x + 10, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Sample count (small, below letter)
        count = sample_counts.get(letter, 0)
        cv2.putText(frame, str(count), (x + 12, y + box_size - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_CYAN, 1)
    
    # Instructions for number keys
    cv2.putText(frame, "Keys: 0-9 for first 10 letters, then A-N for rest",
               (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_YELLOW, 1)
    
    return frame


def draw_status(frame, current_letter, sample_counts, last_save_time, hand_in_zone):
    """Draw status information at the top of the frame."""
    
    # Current letter and count
    count = sample_counts.get(current_letter, 0)
    status_text = f"Recording: {current_letter} | Samples: {count}"
    
    # Status color based on hand position
    if hand_in_zone:
        status_color = COLOR_GREEN
        zone_text = "READY - Press SPACE to save"
    else:
        status_color = COLOR_RED
        zone_text = "Move hand into signing zone"
    
    # Background
    cv2.rectangle(frame, (10, 10), (450, 90), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (450, 90), status_color, 2)
    
    # Text
    cv2.putText(frame, status_text, (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, zone_text, (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    # Last save indicator
    if last_save_time:
        elapsed = (datetime.now() - last_save_time).total_seconds()
        if elapsed < 1.0:
            cv2.putText(frame, "SAVED!", (20, 82),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 2)
    
    # Total samples
    total = sum(sample_counts.values())
    cv2.putText(frame, f"Total: {total}", (380, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    
    return frame


# =============================================================================
# KEY MAPPING
# =============================================================================

def get_letter_from_key(key):
    """
    Map key press to letter index.
    
    Mapping:
        0-9 → letters 0-9 (A-K, skipping index 9 which doesn't exist)
        a-n → letters 10-23
    """
    # Number keys 0-9
    if ord('0') <= key <= ord('9'):
        idx = key - ord('0')
        if idx < len(STATIC_LETTERS):
            return STATIC_LETTERS[idx]
    
    # Letter keys a-n (for indices 10-23)
    if ord('a') <= key <= ord('n'):
        idx = 10 + (key - ord('a'))
        if idx < len(STATIC_LETTERS):
            return STATIC_LETTERS[idx]
    
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main data collection loop."""
    
    print("Initializing detection...")
    mp_resources = detection.initialize_mediapipe()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Load existing data
    df = load_existing_data(OUTPUT_FILE)
    sample_counts = get_sample_counts(df)
    
    # State
    current_letter = STATIC_LETTERS[0]
    mirrored = True
    show_zone = True
    last_save_time = None
    samples_this_session = 0
    
    print("\nReady! Controls:")
    print("  0-9, a-n : Select letter")
    print("  SPACE    : Save sample")
    print("  Z        : Toggle zone display")
    print("  Q        : Quit and save")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if mirrored:
            frame = cv2.flip(frame, 1)
        
        # Process frame (detect face and hands)
        frame, face_refs, hands_data = detection.process_frame(frame, mp_resources)
        
        # Draw face references and zone
        if face_refs:
            frame = detection.draw_face_references(frame, face_refs, show_zone)
        
        # Check if hand is in signing zone
        hand_in_zone = False
        current_hand_data = None
        
        if hands_data:
            for hand_data in hands_data:
                rel_pos = hand_data.get('relative_position')
                if rel_pos and rel_pos['in_signing_zone']:
                    hand_in_zone = True
                    current_hand_data = hand_data
                    break
        
        # Draw UI
        frame = draw_letter_selector(frame, current_letter, sample_counts)
        frame = draw_status(frame, current_letter, sample_counts, last_save_time, hand_in_zone)
        
        cv2.imshow('NGT Data Collection', frame)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('z'):
            show_zone = not show_zone
        
        elif key == ord(' '):  # Spacebar - save sample
            if hand_in_zone and current_hand_data:
                # Get wrist-normalized landmarks (x, y only)
                landmarks_norm = current_hand_data['landmarks_normalized']
                
                # Convert to row
                row = landmarks_to_row(landmarks_norm, current_letter)
                
                # Add to DataFrame
                new_row = pd.DataFrame([row], columns=create_column_names())
                df = pd.concat([df, new_row], ignore_index=True)
                
                # Update counts
                sample_counts[current_letter] = sample_counts.get(current_letter, 0) + 1
                samples_this_session += 1
                last_save_time = datetime.now()
                
                print(f"Saved: {current_letter} (total: {sample_counts[current_letter]})")
            else:
                print("Cannot save: Hand not in signing zone or not detected")
        
        else:
            # Check for letter selection keys
            new_letter = get_letter_from_key(key)
            if new_letter:
                current_letter = new_letter
                print(f"Selected letter: {current_letter}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    mp_resources['hands'].close()
    mp_resources['face_mesh'].close()
    
    # Save data
    if samples_this_session > 0:
        save_data(df, OUTPUT_FILE)
        print(f"\nSession complete: {samples_this_session} new samples collected")
    else:
        print("\nNo new samples collected")
    
    # Print summary
    print("\nSample counts per letter:")
    for letter in STATIC_LETTERS:
        count = sample_counts.get(letter, 0)
        bar = "█" * min(count, 20)
        print(f"  {letter}: {count:3d} {bar}")

# Make file import-friendly

if __name__ == "__main__":
    main()