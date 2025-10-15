import os
import shutil
import pandas as pd

folders_to_process = [
    # data
    {'path': r"data\Raw Audio\Healthy", 'label': "Healthy"},
    {'path': r"data\Raw Audio\Parkinson", 'label': "Parkinson's"},
    {'path': r"data\Raw Audio\Parkinson Dialogue", 'label': "Parkinson's"},
    {'path': r"data\Raw Audio\Parkinson Read", 'label': "Parkinson's"},

    # Parkinson Multi Model DATASET
    {'path': r"Parkinson Multi Model DATASET\Healthy\AUDIO 1 HEALTHY", 'label': "Healthy"},
    {'path': r"Parkinson Multi Model DATASET\Healthy\AUDIO 2 HEALTHY", 'label': "Healthy"},
    {'path': r"Parkinson Multi Model DATASET\Unhealthy\AUDIO 1 UNHEALTHY", 'label': "Parkinson's"},
    {'path': r"Parkinson Multi Model DATASET\Unhealthy\AUDIO 2 UNHEALTHY", 'label': "Parkinson's"},

    # PC-GITA Paths
    {'path': r"PC Gita Vowels\Control\A", 'label': "Healthy"},
    {'path': r"PC Gita Vowels\Control\E", 'label': "Healthy"},
    {'path': r"PC Gita Vowels\Control\I", 'label': "Healthy"},
    {'path': r"PC Gita Vowels\Control\O", 'label': "Healthy"},
    {'path': r"PC Gita Vowels\Control\U", 'label': "Healthy"},
    {'path': r"PC Gita Vowels\Patologicas\A", 'label': "Parkinson's"},
    {'path': r"PC Gita Vowels\Patologicas\E", 'label': "Parkinson's"},
    {'path': r"PC Gita Vowels\Patologicas\I", 'label': "Parkinson's"},
    {'path': r"PC Gita Vowels\Patologicas\O", 'label': "Parkinson's"},
    {'path': r"PC Gita Vowels\Patologicas\U", 'label': "Parkinson's"},
]

# The script will create these in your main project folder.
destination_folder = r"Clean_Audio_Master_Dataset"
output_csv_path = r"master_labels.csv"

audio_extensions = ['.wav', '.mp3', 'flac']


master_list = []
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    print(f"Created destination folder: {destination_folder}")

for item in folders_to_process:
    folder_path = item['path']
    label = item['label']

    if not os.path.isdir(folder_path):
        print(f"[Warning] Path not found, skipping: {folder_path}")
        continue
    
    print(f"Processing: {folder_path} (Label: {label})")
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in audio_extensions):
            source_file_path = os.path.join(folder_path, filename)
            shutil.copy2(source_file_path, destination_folder)
            
            master_list.append({
                'filename': filename,
                'label': label,
                'original_path': source_file_path
            })
            print(f"  Processed: {filename}")

if not master_list:
    print("\n[Error] No audio files were found.")

df = pd.DataFrame(master_list)
df.to_csv(output_csv_path, index=False)

print(f"Total audio files processed: {len(df)}")
print(f"Clean audio files are in: {destination_folder}")
print(f"Master CSV label file is at: {output_csv_path}")
