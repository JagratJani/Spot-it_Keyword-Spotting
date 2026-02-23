import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import sounddevice as sd
import librosa
from python_speech_features import mfcc
import soundfile as sf


def record_audio(duration=2, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    sf.write("first.wav", audio, sample_rate)
    print(f"Recording saved to first.wav")
    print(audio)
    return audio.flatten()

def is_silence(audio, threshold=0.015):
    # checking for silencce detectino 
    rms = np.sqrt(np.mean(audio**2))
    return rms < threshold

def process_audio(audio, sample_rate=16000, check_silence=True):

    # Only check silence during prediction
    if check_silence and is_silence(audio):
        return None
    


    chunks = []
    for i in range(0, len(audio), sample_rate//2):
        chunk = audio[i:i+sample_rate]
        if len(chunk) < sample_rate:
            chunk = np.pad(chunk, (0, sample_rate - len(chunk)), 'constant')
        chunks.append(chunk)

    mfcc_features = []
    for chunk in chunks:
        mfcc_feat = mfcc(chunk, sample_rate, numcep=13, nfilt=26, nfft=512,
                        appendEnergy=True, winfunc=np.hamming)
        delta = librosa.feature.delta(mfcc_feat)
        delta_delta = librosa.feature.delta(mfcc_feat, order=2)
        full_features = np.hstack([mfcc_feat, delta, delta_delta])
        mfcc_features.append(full_features.T)
    
    return np.array(mfcc_features)

def load_dataset(dataset_path, classes):
    X = []
    y = []
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, class_name)
        for file in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file)
            try:
                audio, sr = librosa.load(file_path, sr=16000)
                features = process_audio(audio, check_silence=False)  
                
                if features is not None:
                    for feat in features:
                        X.append(feat)
                        y.append(label_idx)
            except Exception as e:
                print(f"Skipping corrupted file {file_path}: {str(e)}")
    return np.array(X), np.array(y)