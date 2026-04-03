# 🎤 Keyword Spotting System (Speech Commands)

A deep learning-based **Keyword Spotting (KWS)** system built using TensorFlow and MFCC features.  
This project detects spoken keywords from audio input and integrates with a Flask backend for real-time applications.

---

## 🚀 Features

- 🎧 Audio recording and preprocessing using MFCC + delta + delta-delta
- 🧠 CNN-based keyword classification
- 📊 Feature normalization for stable predictions
- 🔁 Chunk-based audio processing
- 🌐 Flask API for real-time keyword detection
- 📦 Supports Google Speech Commands dataset

---

## 🧱 Project Structure

```
Spot-it/
│
├── src/
│   ├── __init__.py
│   ├── audio_processor.py      # Audio recording + MFCC extraction
│   ├── create_model.py         # CNN architecture
│   ├── model_trainer.py        # Training pipeline
│   ├── keyword_spotting.py     # Inference logic
│   └── server.py               # Flask backend
│
├── dataset/                    # Google Speech Commands dataset
├── models/                     # Trained model + normalization stats
├── uploads/                    # Uploaded audio files (Flask)
├── requirements.txt
└── .gitignore
```

---

## 🧠 Model Details

- **Input:** MFCC features (13 + delta + delta-delta = 39 features)
- **Architecture:**
  - 3 × Conv2D + BatchNorm + MaxPooling
  - Dropout (0.25)
  - Dense(128)
  - Softmax output

### Output Classes (12 total):

| Class | Description |
|-------|-------------|
| `_background_noise_` | Silence / background |
| `yes, no, up, down, left, right, on, off, stop, go` | Core keywords |
| `unknown` | All other words grouped |

---

## 📊 Dataset

Uses **Google Speech Commands v0.02**

### Subset Used:

```
yes, no, up, down, left, right, on, off, stop, go
```

- `_background_noise_` → silence
- `unknown` → all other words grouped

---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/JagratJani/Spot-it_Keyword-Spotting.git
cd Spot-it_Keyword-Spotting
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

1. Download the [Google Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)
2. Extract it into:

```
dataset/
```

---

## 🏋️ Train the Model

```bash
python -m src.model_trainer
```

This will:

- Load dataset
- Normalize features
- Train CNN
- Save model to:

```
models/kws_final_1.keras
```

- Save normalization stats:

```
models/feature_mean.npy
models/feature_std.npy
```

---

## 🎯 Run Keyword Detection

```bash
python -m src.keyword_spotting
```

- Records audio (2 seconds)
- Predicts spoken keyword
- Outputs final detection

---

## 🌐 Run Flask Server

```bash
python -m src.server
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload audio file for keyword detection |
| `GET` | `/api/posts` | Get posts (with ads injection) |

---

## 🧪 Example Output

```
Final Detection: stop
```

---

## ⚡ Performance Notes

- Training uses **CPU by default**
- GPU support requires **CUDA / WSL2** setup
- Dataset loading is CPU-bound
- Model inference is lightweight

---

## ⚠️ Known Limitations

- Short words (e.g., `"cat"`) require clearer pronunciation
- Chunk-based processing may miss very brief keywords
- Dataset loading is time-consuming
- Real-time streaming not yet implemented

---

## 🔮 Future Improvements

- 🔄 Real-time streaming keyword detection
- ⚡ GPU acceleration (CUDA / WSL2)
- 📉 Data pipeline optimization (`tf.data`)
- 🎯 Better handling of short keywords
- 📱 TensorFlow Lite conversion (mobile deployment)
- 📡 WebSocket-based live audio processing

---

## 🏁 Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Basic system | ✅ Complete |
| Phase 2 | Inference stabilization | ✅ Complete |
| Phase 3 | Feature normalization | ✅ Complete |
| Phase 4 | Model refinement | ✅ Complete |
| Phase 5 | Multi-keyword training (12-word subset) | ✅ Complete |

---

## 👨‍💻 Author

**Jagrat Jani**

---

## 📄 License

This project is for **educational purposes** only.
