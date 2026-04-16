import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig("../outputs/confusion_matrix.png")
plt.close()

# Feature Importance (for RF)
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    plt.barh(range(len(importances)), importances)
    plt.savefig("../outputs/feature_importance.png")
