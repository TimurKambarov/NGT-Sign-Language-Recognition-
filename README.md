# NGT Sign Language Recognition System

> 🚧 **Work in Progress** — This project is currently under active development

## Project Overview

This project aims to develop a real-time computer vision system for recognizing Dutch Sign Language (Nederlandse Gebarentaal - NGT) fingerspelling. The system captures hand gestures via camera input, processes them using machine learning techniques, and outputs the corresponding letters.

### Goal

Create an accessible learning tool that enables users to practice NGT fingerspelling and receive real-time feedback on their hand movements and positions. This tool is designed to support the approximately 30,000 people in the Netherlands who use NGT as their primary language.

### Key Features

- ✅ Real-time hand detection using MediaPipe Hand Landmarks
- ✅ Face mesh detection for body-relative hand positioning
- ✅ Signing zone validation (ensures hand is at proper height)
- ✅ Data collection module for building training dataset
- ✅ Random Forest classifier with Optuna hyperparameter tuning
- ✅ Real-time letter prediction with stabilization
- 🔲 Dynamic letter support (H, J, Z)
- 🔲 Word building from letters
- 🔲 User-friendly interface

## Repository Structure

```

NGT-Sign-Language-Recognition/
├── data/
│   └── samples.csv              # Collected training data
├── models/
│   └── random_forest.joblib     # Trained model
├── hand_face_detection.py       # Hand + face detection module
├── data_collection.py           # Data collection script
├── train_model.ipynb            # Model training notebook
├── real_time_prediction.py      # Real-time prediction script
├── requirements.txt             # pip dependencies
├── environment.yml              # Conda environment
├── .gitattributes
└── README.md
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/TimurKambarov/NGT-Sign-Language-Recognition.git
cd NGT-Sign-Language-Recognition
```

### 2. Install Dependencies

**Option A: pip (venv)**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Option B: Conda (recommended if pip fails or if you want to train/tune model)**

```bash
conda env create -f environment.yml
conda activate ngt-sign
```

### 3. Collect Training Data

```bash
python data_collection.py
```

Controls:
- `0-9`, `a-n` — Select letter (numbers and letters act as indexes)
- `SPACE` — Save sample
- `q` — Quit and save

### 4. Train the Model

Open `train_model.ipynb` in Jupyter and run all cells, or:

```bash
jupyter notebook train_model.ipynb
```

### 5. Run Real-Time Prediction

```bash
python real_time_prediction.py
```

Controls:
- `q` — Quit
- `m` — Toggle mirror mode
- `z` — Toggle signing zone

## Technical Approach

### Detection Pipeline

The system uses a two-model approach:

| Model | Purpose |
|-------|---------|
| MediaPipe Hands | Detects 21 hand landmarks per hand |
| MediaPipe Face Mesh | Establishes body reference points |

### Why Face Detection?

Proper NGT fingerspelling requires the hand at shoulder/face height. Face detection enables:
- Hand position validation relative to body
- Signing zone enforcement
- Distance-invariant measurements

### Landmark Normalization

| Type | Purpose |
|------|---------|
| Wrist-relative | Hand shape recognition (position independent) |
| Face-relative | Signing zone validation |

### Classification

- **Algorithm:** Random Forest
- **Optimization:** Optuna (100 trials, 5-fold CV, stratified)
- **Features:** 42 (21 landmarks × 2 coordinates)
- **Stabilization:** 10 consecutive matching predictions required
- **Confidence threshold:** 60%

## File Descriptions

### `hand_face_detection.py`

Core detection module that:
- Detects hands and face simultaneously
- Calculates hand position relative to face
- Normalizes landmarks relative to wrist
- Validates signing zone positioning

### `data_collection.py`

Data collection interface that:
- Displays letter selection grid
- Shows real-time sample counts per letter
- Saves landmarks to `data/samples.csv`
- Enforces signing zone for quality data

### `train_model.ipynb`

Jupyter notebook that:
- Loads collected data
- Performs stratified train/test split
- Runs Optuna hyperparameter optimization
- Trains Random Forest classifier
- Evaluates and saves model
- Generates training report

### `real_time_prediction.py`

Real-time prediction script that:
- Loads trained model
- Predicts letters from hand landmarks
- Stabilizes predictions (reduces flickering)
- Displays results on screen and console
- Enforces signing zone

## Configuration

Key parameters in `real_time_prediction.py`:

```python
STABILITY_THRESHOLD = 10    # Frames for stable prediction
CONFIDENCE_THRESHOLD = 0.60 # Minimum confidence (60%)
SIGNING_ZONE = {
    'x_min': -2.5,          # Face widths from nose
    'x_max': 2.5,
    'y_min': -0.7,
    'y_max': 1.7,
}
```

## NGT Alphabet Reference

### Static Letters (Currently Supported)

```
A B C D E F G I K L M N O P Q R S T U V W X Y
```

### Dynamic Letters (Planned)

```
H J Z (require movement)
```

### Hand Landmarks

```
        8   12  16  20      ← Fingertips
        |   |   |   |
        7   11  15  19
        |   |   |   |
        6   10  14  18
        |   |   |   |
    4   5---9---13--17
    |    \         /
    3     \       /
    |      \     /
    2       \   /
    |        \ /
    1         0             ← Wrist
```

## Dependencies

| Library | Purpose |
|---------|---------|
| opencv-python | Webcam, image processing |
| mediapipe | Hand and face detection |
| scikit-learn | Random Forest classifier |
| optuna | Hyperparameter optimization |
| numpy, pandas | Data handling |
| joblib | Model serialization |
| fastdtw | Dynamic letters (planned) |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Metadata error with pip | Use Conda: `conda env create -f environment.yml` |
| `conda` not found | Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) |
| MediaPipe fails | Use Python 3.8-3.11 (not 3.12+) |
| Webcam not detected | Check system camera permissions |
| Model not found | Run `train_model.ipynb` first |
| Low accuracy | Collect more training samples per letter |
|  Kernel fails (jupyter notebook) | Use Conda: `conda install -n ngt-sign  ipykernel --update-deps --force-reinstall` |

## Roadmap

- [x] Hand detection with MediaPipe
- [x] Face mesh for body-relative positioning
- [x] Signing zone validation
- [x] Data collection interface
- [x] Model training pipeline with Optuna
- [x] Real-time letter prediction
- [ ] Dynamic letter support (H, J, Z)
- [ ] Word building from letters
- [ ] Left-hand support (mirroring)
- [ ] UI
- [ ] Feedback system for learning

## References

- [NGT Alphabet Video](https://youtu.be/oZMyER7fWJY?si=HOpoH6l4ULbe_58W)
- [NGT Signbank](https://signbank.cls.ru.nl/datasets/NGT/)

## Acknowledgments

- **Nienke Fluitman** — NGT Teacher and project stakeholder
- **Irene van Blerck & Karna Rewatkar** — Project supervisors
- **Breda University of Applied Sciences** — ADS&AI Program

## License

*To be determined*

---

*Part of the ADS&AI Sign Language Challenge at Breda University of Applied Sciences*