import numpy as np

data1 = np.load('data/hand_gesture_data.npy', allow_pickle=True).item()
data2 = np.load('data/hand_gesture_data_new.npy', allow_pickle=True).item()

keypoints1 = data1['keypoints']
labels1 = data1['labels']

keypoints2 = data2['keypoints']
labels2 = data2['labels']

combined_keypoints = np.concatenate((keypoints1, keypoints2), axis=0)
combined_labels = np.concatenate((labels1, labels2), axis=0)

combined_data = {
    'keypoints': combined_keypoints,
    'labels': combined_labels
}

np.save('data/hand_gesture_data_combined.npy', combined_data)

print(f"Shape of combined keypoints: {combined_keypoints.shape}")
print(f"Shape of combined labels: {combined_labels.shape}")
