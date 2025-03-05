import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")


df = pd.read_csv("/content/processed_spotify_data.csv")


sns.set_style("darkgrid")


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(x=df['mood'].value_counts().index,
            y=df['mood'].value_counts().values,
            palette="viridis", ax=axes[0])
axes[0].set_title("Mood Distribution in Spotify Dataset")
axes[0].set_xlabel("Mood")
axes[0].set_ylabel("Count")


sns.scatterplot(x=df['valence'], y=df['energy'], hue=df['mood'], palette="deep", alpha=0.7, ax=axes[1])
axes[1].set_title("Mood Classification based on Valence & Energy")
axes[1].set_xlabel("Valence (Happiness)")
axes[1].set_ylabel("Energy (Liveliness)")


plt.tight_layout()
plt.show()


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC curve
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green', 'orange']  # Adjust colors as needed
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {rf_model.classes_[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Multiclass)')
plt.legend(loc='lower right')
plt.show()

# Feature Distribution Plots
print('\n')
plt.figure(figsize=(12, 6))
df.drop(columns=['mood']).hist(bins=30, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle("Audio Feature Distributions", fontsize=16)
plt.show()

#  Feature Importance
feature_importances = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=X.columns[:len(feature_importances)], palette='coolwarm')
plt.title("Feature Importance")
plt.show()

