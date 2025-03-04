import numpy as np

data = np.load('combined_hand_gesture_data.npy', allow_pickle=True).item()  # Use .item() to get the dict

keypoints = data['keypoints']
labels = data['labels']

mask = labels != 'thumbs_up'

filtered_keypoints = keypoints[mask]
filtered_labels = labels[mask]

filtered_data = {
    'keypoints': filtered_keypoints,
    'labels': filtered_labels
}

np.save('filtered_hand_gesture_data.npy', filtered_data)

print("Filtered data saved to 'filtered_hand_gesture_data.npy'")
