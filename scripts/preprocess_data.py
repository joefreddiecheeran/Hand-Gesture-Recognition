import numpy as np
from sklearn.model_selection import train_test_split

data = np.load("data/hand_gesture_data.npy", allow_pickle=True).item()

X = np.array([sample for sample in data['keypoints']])
y = np.array([label for label in data['labels']])

X = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

print("Data preprocessing complete.")
