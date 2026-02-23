import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.audio_processor import load_dataset
from src.create_model import create_model
import tensorflow as tf

DATASET_PATH = "dataset"
CLASSES = ['_background_noise_', 'cat', 'dog', 'house']
MODEL_PATH = "models/kws_final_1.h5"

os.makedirs("models", exist_ok=True)
print("Loading dataset...")
X, y = load_dataset(DATASET_PATH, CLASSES)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#  model creatiing and trainnig
model = create_model(input_shape=X_train[0].shape)
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,  
    batch_size=32,
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]
)

print(f"Model saved to {MODEL_PATH}")