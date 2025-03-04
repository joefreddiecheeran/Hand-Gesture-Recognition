import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train_encoded)

y_pred = rf_model.predict(X_test)

print("Random Forest Accuracy: ", accuracy_score(y_test_encoded, y_pred))
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

# conf_matrix = confusion_matrix(y_test_encoded, y_pred)

# plt.figure(figsize=(10, 7))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.show()

joblib.dump(rf_model, 'models/random_forest_model.joblib')

np.save('models/label_encoder.npy', label_encoder.classes_)

print("Model and label encoder saved successfully.")
