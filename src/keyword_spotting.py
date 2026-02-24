import numpy as np
from src.audio_processor import record_audio, process_audio
import os
from tensorflow.keras.models import load_model


class KeywordSpotter:
    def __init__(self, model_path='models/kws_final_1.keras'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Train first!")
            
        self.model = load_model(model_path, compile=False)
        self.mean = np.load("models/feature_mean.npy")
        self.std = np.load("models/feature_std.npy")
        self.labels = ['_background_noise_', 'cat', 'dog', 'house']
        self.threshold = 0.75  
        self.min_activations = 3  

    def predict_keyword(self, audio_features):
        if audio_features is None:
            return ['silence']
        
        # Normalize the audio features using the mean and std from training
        audio_features = (audio_features - self.mean) / (self.std + 1e-6)
        
        predictions = self.model.predict(audio_features)
        results = []
        for pred in predictions:
            max_index = np.argmax(pred)
            if pred[max_index] > self.threshold and self.labels[max_index] != '_background_noise_':
                results.append(self.labels[max_index])
            else:
                results.append('_background_noise_')
        # print("results", results)
        return results
    
    def analyze_audio(self, audio):
        features = process_audio(audio)
        if features is not None:
            features = (features - self.mean) / self.std
        predictions = self.predict_keyword(features)
        
        keyword_counts = {}
        for pred in predictions:
            if pred not in ['_background_noise_', 'silence']:
                keyword_counts[pred] = keyword_counts.get(pred, 0) + 1
        # print("keyword counts: ", keyword_counts)
        
        for keyword, count in keyword_counts.items():
            if count >= self.min_activations:
                # print("hello", max(keyword_counts, key=keyword_counts.get))
                # return keyword
                return max(keyword_counts, key=keyword_counts.get)
        # print(keyword_counts)
        return '_background_noise_'

if __name__ == "__main__":
    try:
        audio = record_audio()
        spotter = KeywordSpotter()
        prediction = spotter.analyze_audio(audio)
        print(f"Final Detection: {prediction}")
    except FileNotFoundError as e:
        print("Error:", str(e))
        print("Solution: First train the model using 'python model_trainer.py'")
