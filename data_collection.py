import cv2
import numpy as np
import pandas as pd
import os
import time
import hand_face_detection as detection
from guide import show_guide



# Configuration

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "samples.csv")

FRAMES_STATIC = 30
FRAMES_DYNAMIC = 75

STATIC_LETTERS = list("ABCDEFGIKLMNOPQRSTUVWXY")
DYNAMIC_LETTERS = ['H', 'J', 'Z']
ALL_LETTERS = sorted(STATIC_LETTERS + DYNAMIC_LETTERS)

CONTINUOUS_SAMPLES_STATIC = 50
CONTINUOUS_SAMPLES_DYNAMIC = 100
CONTINUOUS_COUNTDOWN = 3
CONTINUOUS_PAUSE = 1.0

LETTERS_PER_ROW = 9

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_BLUE = (255, 0, 0)


# Data Management

def create_column_names():
    """
    Create column names for the CSV
    Format: sample_id, frame_id, wrist_abs_x, wrist_abs_y, wrist_abs_z,
            x0, y0, z0, x1, y1, z1, ..., x20, y20, z20, label
    Total: 69 columns
    """
    columns = ['sample_id', 'frame_id', 'wrist_abs_x', 'wrist_abs_y', 'wrist_abs_z']
    for i in range(21):
        columns.append(f"x{i}")
        columns.append(f"y{i}")
        columns.append(f"z{i}")
    columns.append("label")
    return columns


def landmarks_to_row(landmarks_normalized, landmarks_absolute, sample_id, frame_id, label):
    """
    Convert landmarks to a flat row for CSV
    
    Args:
        landmarks_normalized: List of 21 dicts with normalized x, y, z (relative to wrist)
        landmarks_absolute: List of 21 dicts with absolute x, y, z (original values)
        sample_id: Unique identifier for this sample
        frame_id: Frame number within the sample
        label: Letter label
    
    Returns:
        List: [sample_id, frame_id, wrist_abs_x, wrist_abs_y, wrist_abs_z, 
               x0, y0, z0, ..., x20, y20, z20, label]
    """
    row = [sample_id, frame_id]
    
    # Add absolute wrist position (landmark 0 before normalization)
    wrist_abs = landmarks_absolute[0]
    row.extend([wrist_abs['x'], wrist_abs['y'], wrist_abs['z']])
    
    # Add normalized landmarks
    for lm in landmarks_normalized:
        row.extend([lm['x'], lm['y'], lm['z']])
    
    row.append(label)
    return row


def load_existing_data(filepath):
    """Load existing CSV data or create empty DataFrame"""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} existing rows from {filepath}")
        return df
    else:
        columns = create_column_names()
        print("Starting with empty dataset")
        return pd.DataFrame(columns=columns)


def save_data(df, filepath):
    """Save DataFrame to CSV"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} rows to {filepath}")


def get_next_sample_id(df):
    """Get the next available sample_id"""
    if len(df) == 0:
        return 0
    return df['sample_id'].max() + 1


def get_sample_counts(df):
    """Get count of complete samples per letter"""
    if len(df) == 0:
        return {}
    samples_per_label = df.groupby('label')['sample_id'].nunique()
    return samples_per_label.to_dict()


def reset_dataset(filepath):
    """Delete existing dataset and create empty one"""
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Deleted {filepath}")
    columns = create_column_names()
    df = pd.DataFrame(columns=columns)
    return df

def draw_letter_selector(frame, current_letter, sample_counts):
    """Draw letter selection UI"""
    h, w, _ = frame.shape
    
    # Background panel
    panel_height = 200
    cv2.rectangle(frame, (10, h - panel_height - 10), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, h - panel_height - 10), (w - 10, h - 10), COLOR_WHITE, 2)
    
    # Title with controls
    cv2.putText(frame, "Letter | SPACE=record | 1=quit | 2=reset | 3=guide | 4=continuous",
               (20, h - panel_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)
    
    # Draw letters in grid
    start_y = h - panel_height + 45
    box_size = 32
    margin = 4
    
    for idx, letter in enumerate(ALL_LETTERS):
        row = idx // LETTERS_PER_ROW
        col = idx % LETTERS_PER_ROW
        
        x = 20 + col * (box_size + margin)
        y = start_y + row * (box_size + margin)
        
        # Highlight selected letter
        if letter == current_letter:
            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), COLOR_GREEN, -1)
            text_color = (0, 0, 0)
        else:
            # Color code: blue for dynamic, white for static
            border_color = COLOR_BLUE if letter in DYNAMIC_LETTERS else COLOR_WHITE
            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), border_color, 1)
            text_color = COLOR_WHITE
        
        # Letter
        cv2.putText(frame, letter, (x + 8, y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Sample count
        count = sample_counts.get(letter, 0)
        cv2.putText(frame, str(count), (x + 10, y + box_size - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_CYAN, 1)
    
    return frame


def draw_status(frame, current_letter, sample_counts, is_recording, frames_recorded, 
                total_frames, continuous_mode=False, continuous_count=0):
    """Draw status information"""
    h, w, _ = frame.shape
    
    count = sample_counts.get(current_letter, 0)
    is_dynamic = current_letter in DYNAMIC_LETTERS
    
    if is_recording:
        status_color = COLOR_ORANGE
        if continuous_mode:
            status_text = f"CONTINUOUS: {current_letter} | Sample {continuous_count}"
        else:
            status_text = f"RECORDING: {current_letter} | Frame: {frames_recorded}/{total_frames}"
        zone_text = "Hold steady..." if not is_dynamic else "Perform gesture..."
        
        progress = frames_recorded / total_frames
        bar_width = 400
        cv2.rectangle(frame, (20, 75), (20 + bar_width, 90), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 75), (20 + int(bar_width * progress), 90), COLOR_ORANGE, -1)
    else:
        gesture_type = "DYNAMIC" if is_dynamic else "STATIC"
        frames_info = f"{FRAMES_DYNAMIC}f" if is_dynamic else f"{FRAMES_STATIC}f"
        status_text = f"Selected: {current_letter} [{gesture_type}] | Samples: {count} | Frames: {frames_info}"
        zone_text = "Press SPACE to record | Press 4 for continuous"
        status_color = COLOR_GREEN
    
    cv2.rectangle(frame, (10, 10), (650, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (650, 100), status_color, 2)
    
    cv2.putText(frame, status_text, (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
    cv2.putText(frame, zone_text, (20, 58),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    total = sum(sample_counts.values())
    cv2.putText(frame, f"Total: {total}", (560, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    
    return frame


def draw_pause_message(frame, completed, total):
    """Draw pause message between continuous recordings"""
    h, w, _ = frame.shape
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
    cv2.rectangle(frame, (w//4, h//3), (3*w//4, 2*h//3), COLOR_GREEN, 3)
    
    msg1 = f"Completed {completed} out of {total}"
    msg2 = "Next recording starting..."
    msg3 = "Press '1' to stop"
    
    cv2.putText(frame, msg1, (w//4 + 40, h//2 - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_GREEN, 2)
    cv2.putText(frame, msg2, (w//4 + 60, h//2 + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    cv2.putText(frame, msg3, (w//4 + 80, h//2 + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1)
    
    return frame


def draw_countdown(frame, count):
    """Draw countdown overlay"""
    h, w, _ = frame.shape
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
    
    if count > 0:
        cv2.putText(frame, str(count), (w//2 - 60, h//2 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 6, COLOR_YELLOW, 12)
    else:
        cv2.putText(frame, "GO!", (w//2 - 80, h//2 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 4, COLOR_GREEN, 10)
    
    return frame
    """Draw countdown overlay"""
    h, w, _ = frame.shape
    
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
    
    # Countdown number
    if count > 0:
        cv2.putText(frame, str(count), (w//2 - 60, h//2 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 6, COLOR_YELLOW, 12)
    else:
        cv2.putText(frame, "GO!", (w//2 - 80, h//2 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 4, COLOR_GREEN, 10)
    
    return frame


def get_letter_from_key(key):
    """Map key press to letter"""
    try:
        char = chr(key).upper()
    except:
        return None
    
    if char in ALL_LETTERS:
        return char
    
    return None

def record_single_sample(current_letter, next_sample_id, df, mp_resources, cap, mirrored):
    """
    Record a single sample
    Returns: updated df, next_sample_id, success (bool)
    """
    is_dynamic = current_letter in DYNAMIC_LETTERS
    total_frames = FRAMES_DYNAMIC if is_dynamic else FRAMES_STATIC
    
    recording_frames = []
    frames_recorded = 0
    
    print(f"\nRecording {current_letter} ({total_frames} frames)...")
    
    while frames_recorded < total_frames:
        ret, frame = cap.read()
        if not ret:
            return df, next_sample_id, False
        
        if mirrored:
            frame = cv2.flip(frame, 1)
        
        # Process frame
        frame, face_refs, hands_data = detection.process_frame(frame, mp_resources)
        
        # Check if hand detected
        if hands_data and face_refs:
            hand_data = hands_data[0]
            
            # Store both normalized and absolute landmarks
            landmarks_norm = hand_data['landmarks_normalized']
            landmarks_abs = hand_data['landmarks']
            
            recording_frames.append((landmarks_norm, landmarks_abs))
            frames_recorded = len(recording_frames)
            
            # Draw recording status
            frame = draw_status(frame, current_letter, {}, True, 
                              frames_recorded, total_frames)
        else:
            # No hand detected - show warning
            cv2.putText(frame, "HAND NOT DETECTED!", (400, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_RED, 3)
        
        cv2.imshow('NGT Data Collection', frame)
        
        # Allow quit during recording
        if cv2.waitKey(1) & 0xFF == ord('1'):
            print("Recording cancelled")
            return df, next_sample_id, False
    
    # Save all frames
    rows = []
    for frame_id, (lm_norm, lm_abs) in enumerate(recording_frames):
        row = landmarks_to_row(lm_norm, lm_abs, next_sample_id, frame_id, current_letter)
        rows.append(row)
    
    # Add to DataFrame
    new_rows = pd.DataFrame(rows, columns=create_column_names())
    df = pd.concat([df, new_rows], ignore_index=True)
    
    print(f"✓ Saved: {current_letter} (sample_id: {next_sample_id})")
    
    return df, next_sample_id + 1, True


def record_continuous(current_letter, next_sample_id, df, sample_counts, 
                     mp_resources, cap, mirrored):
    """
    Record multiple samples continuously with countdown
    """
    is_dynamic = current_letter in DYNAMIC_LETTERS
    total_frames = FRAMES_DYNAMIC if is_dynamic else FRAMES_STATIC
    total_samples = CONTINUOUS_SAMPLES_DYNAMIC if is_dynamic else CONTINUOUS_SAMPLES_STATIC
    
    print(f"\n{'='*60}")
    print(f"CONTINUOUS RECORDING MODE")
    print(f"Letter: {current_letter} ({'DYNAMIC' if is_dynamic else 'STATIC'})")
    print(f"Samples to record: {total_samples}")
    print(f"Frames per sample: {total_frames}")
    print(f"Press '1' during recording to stop early")
    print(f"{'='*60}\n")
    
    samples_recorded = 0
    continuous_data = []  # Store samples temporarily
    stopped_early = False
    
    for sample_num in range(total_samples):
        print(f"\nSample {sample_num + 1}/{total_samples}")
        
        # Countdown
        for countdown in range(CONTINUOUS_COUNTDOWN, 0, -1):
            ret, frame = cap.read()
            if not ret:
                break
            
            if mirrored:
                frame = cv2.flip(frame, 1)
            
            frame, _, _ = detection.process_frame(frame, mp_resources)
            frame = draw_countdown(frame, countdown)
            
            cv2.imshow('NGT Data Collection', frame)
            
            # Check for stop during countdown
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('1'):
                stopped_early = True
                break
        
        if stopped_early:
            break
        
        # Show "GO!" briefly
        ret, frame = cap.read()
        if mirrored:
            frame = cv2.flip(frame, 1)
        frame, _, _ = detection.process_frame(frame, mp_resources)
        frame = draw_countdown(frame, 0)
        cv2.imshow('NGT Data Collection', frame)
        cv2.waitKey(300)
        
        temp_df, temp_sample_id, success = record_single_sample(
            current_letter, next_sample_id + samples_recorded, 
            pd.DataFrame(columns=create_column_names()),
            mp_resources, cap, mirrored
        )
        
        if success:
            continuous_data.append(temp_df)
            samples_recorded += 1
        
        # Check for stop key during recording
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            stopped_early = True
            break
        
        # Pause between samples (except last one)
        if sample_num < total_samples - 1 and not stopped_early:
            # Show completion message during pause
            pause_start = time.time()
            while time.time() - pause_start < CONTINUOUS_PAUSE:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if mirrored:
                    frame = cv2.flip(frame, 1)
                
                frame, _, _ = detection.process_frame(frame, mp_resources)
                frame = draw_pause_message(frame, samples_recorded, total_samples)
                
                cv2.imshow('NGT Data Collection', frame)
                
                # Check for stop during pause
                key = cv2.waitKey(30) & 0xFF
                if key == ord('1'):
                    stopped_early = True
                    break
            
            if stopped_early:
                break
    
    # Handle early stop
    if stopped_early:
        print("\nWhat would you like to do?")
        print("  1 - SAVE all recorded samples")
        print("  2 - DISCARD all recorded samples")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            print(f"Saving {samples_recorded} samples...")
            # Add all temporary data to main dataframe
            for temp_df in continuous_data:
                df = pd.concat([df, temp_df], ignore_index=True)
            
            sample_counts[current_letter] = sample_counts.get(current_letter, 0) + samples_recorded
            next_sample_id += samples_recorded
            print(f"Saved {samples_recorded} samples for letter {current_letter}")
        elif choice == '2':
            print(f"Discarding {samples_recorded} samples...")
            continuous_data = []
            print("All continuous recordings discarded")
        else:
            print("Invalid choice, hence discarding recordings.")
            continuous_data = []
    else:
        # Completed all samples normally
        print(f"\nContinuous recording complete: {samples_recorded} samples recorded")
        
        # Add all temporary data to main dataframe
        for temp_df in continuous_data:
            df = pd.concat([df, temp_df], ignore_index=True)
        
        sample_counts[current_letter] = sample_counts.get(current_letter, 0) + samples_recorded
        next_sample_id += samples_recorded
    
    return df, next_sample_id, sample_counts


def main():
    """Main data collection loop"""
    
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Static letters: {', '.join(STATIC_LETTERS)} ({FRAMES_STATIC} frames)")
    print(f"Dynamic letters: {', '.join(DYNAMIC_LETTERS)} ({FRAMES_DYNAMIC} frames)")
    
    print("\nInitializing detection...")
    mp_resources = detection.initialize_mediapipe()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    df = load_existing_data(OUTPUT_FILE)
    sample_counts = get_sample_counts(df)
    next_sample_id = get_next_sample_id(df)
    
    current_letter = ALL_LETTERS[0]
    mirrored = True
    is_recording = False
    
    print("\nReady! Controls:")
    print("  A-Z      : Select letter")
    print("  SPACE    : Record single sample")
    print("  1        : Quit and save")
    print("  2        : Reset dataset")
    print("  3        : Show guide")
    print("  4        : Continuous recording (50 static / 100 dynamic)")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if mirrored:
            frame = cv2.flip(frame, 1)
        
        frame, face_refs, hands_data = detection.process_frame(frame, mp_resources)
        
        frame = draw_letter_selector(frame, current_letter, sample_counts)
        frame = draw_status(frame, current_letter, sample_counts, False, 0, 0)
        
        if hands_data:
            center = hands_data[0]['center_px']
            cv2.circle(frame, center, 10, COLOR_GREEN, -1)
        
        cv2.imshow('NGT Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('1'):
            break
        
        elif key == ord('2'):
            print("\nWARNING: This will delete all collected data!")
            confirmation = input("Type 'reset' to confirm: ")
            if confirmation.strip().lower() == 'reset':
                df = reset_dataset(OUTPUT_FILE)
                sample_counts = {}
                next_sample_id = 0
                print("Dataset reset successfully!")
            else:
                print("Reset cancelled.")
        
        elif key == ord('3'):
            print(f"Showing guide for: {current_letter}")
            show_guide(current_letter)
        
        elif key == ord('4'):
            print(f"Starting continuous recording for letter: {current_letter}")
            df, next_sample_id, sample_counts = record_continuous(
                current_letter, next_sample_id, df, sample_counts,
                mp_resources, cap, mirrored
            )
        
        elif key == ord(' '):
            df, next_sample_id, success = record_single_sample(
                current_letter, next_sample_id, df, mp_resources, cap, mirrored
            )
            if success:
                sample_counts[current_letter] = sample_counts.get(current_letter, 0) + 1
        
        else:
            new_letter = get_letter_from_key(key)
            if new_letter:
                current_letter = new_letter
                print(f"Selected letter: {current_letter} ({'DYNAMIC' if new_letter in DYNAMIC_LETTERS else 'STATIC'})")
    
    cap.release()
    cv2.destroyAllWindows()
    mp_resources['hands'].close()
    mp_resources['face_mesh'].close()
    
    save_data(df, OUTPUT_FILE)
    
    print("\nSession complete")
    print("\nSample counts per letter:")
    for letter in ALL_LETTERS:
        count = sample_counts.get(letter, 0)
        letter_type = "DYN" if letter in DYNAMIC_LETTERS else "STA"
        bar = "█" * min(count, 20)
        print(f"  {letter} [{letter_type}]: {count:3d} {bar}")
    
    total_rows = len(df)
    total_samples = sum(sample_counts.values())
    print(f"\nTotal: {total_samples} samples ({total_rows} rows)")


if __name__ == "__main__":
    main()