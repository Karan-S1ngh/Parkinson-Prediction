import pandas as pd
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import os
from tqdm import tqdm

def extract_all_features(file_path):
    # Initialize all features to NaN
    # --- CHANGE: Initialized 40 MFCCs ---
    features = {f'mfcc_{i+1}': np.nan for i in range(40)} 
    
    # Add other feature keys
    extra_features = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                      'rms_energy', 'mean_pitch', 'jitter_local', 'shimmer_local', 'hnr']
    for f in extra_features:
        features[f] = np.nan

    # Librosa Features
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # --- CHANGE: Set n_mfcc=40 ---
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        
        # --- CHANGE: Loop for 40 MFCCs ---
        for i in range(40):
            features[f'mfcc_{i+1}'] = mfccs[i]

        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features['rms_energy'] = np.mean(librosa.feature.rms(y=y))
    except Exception as e:
        print(f"Could not process Librosa features for {os.path.basename(file_path)}: {e}")

    # Parselmouth Features (No changes here)
    try:
        sound = parselmouth.Sound(file_path)
        pitch = sound.to_pitch()
        
        features['mean_pitch'] = call(pitch, "Get mean", 0, 0, "Hertz")
        
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        n_pulses = call(point_process, "Get number of points")
        if n_pulses >= 2:
            features['jitter_local'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features['shimmer_local'] = call(point_process, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features['hnr'] = call(harmonicity, "Get mean", 0, 0)
    except Exception as e:
        # Pass silently on Parselmouth errors to avoid clutter
        pass

    return features

# --- Main Script Logic (Output filename changed) ---

input_csv_path = 'master_labels.csv'
df = pd.read_csv(input_csv_path)
all_features = []

print("Starting feature extraction for 40 MFCCs...")
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    file_path = row['original_path']
    features = extract_all_features(file_path)
    all_features.append(features)

features_df = pd.DataFrame(all_features)
final_df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

# --- CHANGE: New output filename ---
output_csv_path = 'features_dataset_40mfcc.csv'
final_df.to_csv(output_csv_path, index=False)

print(f"\nFeature extraction complete! The final dataset is saved as '{output_csv_path}'")