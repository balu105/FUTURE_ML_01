import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("/content/processed_spotify_data.csv")

columns_to_drop = ['Unnamed: 0', 'track_id', 'album_name', 'track_name', 'artists']
df = df.drop(columns=columns_to_drop, errors='ignore')

df.dropna(inplace=True)


numerical_features = df.select_dtypes(include=np.number).columns
corr_matrix = df[numerical_features].corr()

high_corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.85:  
            colname = corr_matrix.columns[i]
            high_corr_features.add(colname)


df.drop(columns=high_corr_features, inplace=True, errors='ignore') 
if 'mood' in df.columns:
    X = df.drop(columns=['mood'])
    y = df['mood']
else:
    raise KeyError("The 'mood' column is not found in the DataFrame. Please check your data or previous steps.")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=0.95)  
X_pca = pca.fit_transform(X_scaled)


X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=3, random_state=42)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n \t \t \t", report)
print("Confusion Matrix:\n", conf_matrix)


# Correlation heatmap to show feature relationships
plt.figure(figsize=(12, 8))
corr = df[selected_features].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True)
plt.title("Feature Correlation Heatmap")
plt.show()