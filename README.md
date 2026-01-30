# NGT Sign Language Recognition

A real-time sign language recognition system that uses computer vision and machine learning to recognize Nederlandse Gebarentaal (Dutch Sign Language) fingerspelling alphabet gestures.

## Project Overview

This project implements an automated sign language recognition system capable of interpreting hand gestures for all 26 letters of the alphabet in real-time. The system uses MediaPipe for hand detection and landmark extraction, combined with a Random Forest classifier for gesture classification.

### Key Features

- **Real-time Recognition**: Live webcam-based hand gesture detection and classification
- **Complete Alphabet Support**: Recognizes all 26 letters (A-Z) of the NGT fingerspelling alphabet
- **Robust Hand Tracking**: Uses MediaPipe for hand landmark detection
- **Intelligent Zone Detection**: Defines optimal signing area relative to face position
- **Interactive Interface**: Real-time visual feedback with confidence scores and status indicators

## Technical Architecture

### Components

1. **Hand and Face Detection** (`hand_face_detection.py`)
   - Coordinate normalization and feature preparation

2. **Model Training** (`train_model.ipynb`)
   - Train and evaluate Random Forest Classifier model

3. **Data Collection** (`data_collection.py`)
   - Collect all necessary hand-gestures data

4. **Real-time Prediction (no UI)** (`real_time_prediction.py`)
   - Live webcam gesture recognition

5. **Real-time Prediction (with UI)** (`real_time_prediction.py`)
   - Same as above, but with user interface

### Model Performance

- **Algorithm**: Random Forest Classifier
- **Features**: 66-dimensional feature vector (63 normalized landmarks + 3 absolute wrist coordinates)
- **Accuracy**: >95% on test dataset, but lower during actual real-time detection
- **Training Data**: Individual frame-based classification approach
- **Optimization**: Hyperparameter tuning with Optuna (5-fold cross-validation)

## Installation & Setup

### Prerequisites

- Recommended Python 3.10 or higher
- Webcam for real-time recognition

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TimurKambarov/NGT-Sign-Language-Recognition.git
   cd NGT-Sign-Language-Recognition
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

1. **Prepare training data**:
   ```bash
   python data_collection.py
   ```

2. **Train the model**:
   ```bash
   jupyter notebook train_model.ipynb
   ```

### Real-time Recognition

Run the real-time recognition system:

```bash
python real_time_prediction.py
```

### Controls

- **Q**: Quit application
- **M**: Toggle mirror mode
- **Z**: Toggle signing zone visibility


## File Structure

```
NGT-Sign-Language-Recognition-/
├── data/
│   NGT_gestures/                # Hand gestures pictures
│   └── samples.csv              # Training dataset
├── models/
│   ├── random_forest_model.joblib    # Trained RF model
│   └── label_encoder_rf.pkl          # Label encoder
├── hand_face_detection.py       # MediaPipe detection utilities
├── real_time_prediction.py      # Main recognition application
├── train_model.ipynb           # Model training notebook
├── streamlit_app.py            # Data collection tool
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Performance Metrics

The system achieves the following performance characteristics:

- **Accuracy**: >95% on test dataset, but ~40% confidence in real-time prediction
- **Recognition Latency**: <100ms per prediction
- **Supported Gestures**: All 26 letters (A-Z)

## Technical Details

### Feature Engineering

The system uses a 66-dimensional feature vector consisting of:
- **63 normalized landmarks**: Hand keypoints relative to wrist position (21 points × 3 coordinates)
- **3 absolute coordinates**: Wrist position for spatial context

### Model Architecture

- **Algorithm**: Random Forest with optimized hyperparameters
- **Training Strategy**: Per-frame classification approach
- **Validation**: 5-fold cross-validation during hyperparameter tuning
- **Optimization**: Optuna-based parameter search

### Recognition Pipeline

1. **Frame Capture**: Webcam video input
2. **Face Detection**: Establish reference coordinate system
3. **Hand Detection**: Extract 21 hand landmarks
4. **Feature Extraction**: Normalize and prepare feature vector
5. **Classification**: Random Forest prediction with confidence score
6. **Post-processing**: Confidence filtering and temporal smoothing

## Limitations

### Current Model Constraints

This implementation serves as a proof-of-concept and has several important limitations:

- **Limited Training Data**: The current model was trained on a relatively small dataset, which significantly impacts recognition accuracy
- **Low Recognition Confidence**: Due to insufficient training samples, the model typically achieves confidence scores around 40% or lower
- **Reduced Dynamic Gesture Performance**: Dynamic gestures (letters requiring movement) show even lower confidence scores compared to static hand positions
- **Single User Bias**: Training data may be biased toward specific hand shapes, sizes, or signing styles
- **Environmental Sensitivity**: Performance may vary significantly under different lighting conditions or camera angles

### Recommended Improvements

To achieve production-ready performance, the following enhancements are needed:
- Collect substantially more training data (1000+ samples per letter from multiple users)
- Include diverse hand sizes and signing styles in training data
- Implement temporal smoothing and sequence-based recognition for dynamic gestures
- Add data augmentation techniques to improve model robustness
- Consider deep learning approaches for better feature extraction

**Note**: The current confidence threshold is set to 55% for display purposes, but actual predictions often fall below this threshold due to the limited training data.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Future Enhancements

- Support for word-level recognition
- Web application development
- Extended vocabulary beyond fingerspelling
- Multi-hand gesture recognition