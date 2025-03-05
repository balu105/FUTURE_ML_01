import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Load dataset safely
file_path = "/content/dataset.csv"  
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: The file '{file_path}' does not exist!")

df = pd.read_csv(file_path)

print(" Original Dataset Sample:")
print(df.head())


columns_to_drop = ['Unnamed: 0', 'track_id', 'album_name', 'track_name', 'artists']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')


df = df.dropna()

if 'valence' not in df.columns or 'energy' not in df.columns:
    raise ValueError("Error: Required columns 'valence' and 'energy' are missing from dataset.")


def classify_mood(valence, energy):
    if valence > 0.6 and energy > 0.5:
        return 'Happy'
    elif valence <= 0.6 and energy > 0.5:
        return 'Energetic'
    elif valence <= 0.5 and energy <= 0.5:
        return 'Sad'
    else:
        return 'Calm'


df['mood'] = df.apply(lambda row: classify_mood(row['valence'], row['energy']), axis=1)


features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']

missing_features = [feat for feat in features if feat not in df.columns]
if missing_features:
    raise ValueError(f"Error: Missing required feature columns: {missing_features}")

X = df[features]
y = df['mood']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_scaled_df = pd.DataFrame(X_scaled, columns=features)

df_processed = pd.concat([X_scaled_df, y], axis=1)
df_processed.to_csv("processed_spotify_data.csv", index=False)

print("\nData Preprocessing Complete!")
print(" Preprocessed dataset saved as 'processed_spotify_data.csv'.")
print("\n Mood Distribution:\n", df['mood'].value_counts())