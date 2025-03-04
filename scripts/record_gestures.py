import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2,  
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

all_keypoints = []
all_labels = []

GESTURE_LABELS = {
    '1': 'ThumbsUp',
    'a': 'A',
    'b': 'B',
    'c': 'C',
    'd': 'D',
    'e': 'E',
    'f': 'F',
    'g': 'G',
    'h': 'H',
    'i': 'I',
    'j': 'J',
    'k': 'K',
    'l': 'L',
    'm': 'M',
    'n': 'N',
    'o': 'O',
    'p': 'P',
    'q': 'Q',
    'r': 'R',
    's': 'S',
    't': 'T',
    'u': 'U',
    'v': 'V',
    'w': 'W',
    'x': 'X',
    'y': 'Y',
    'z': 'Z'
}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        keypoints = [] 
        for hand_landmarks in results.multi_hand_landmarks:
            hand_keypoints = []
            for landmark in hand_landmarks.landmark:
                hand_keypoints.append([landmark.x, landmark.y, landmark.z])
            keypoints.append(np.array(hand_keypoints).flatten())

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(keypoints) == 2:
            combined_keypoints = np.concatenate((keypoints[0], keypoints[1]))
        else:
            combined_keypoints = np.concatenate((keypoints[0], np.zeros(21*3)))

        cv2.imshow("Hand Gesture Recorder", image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('0'):  # Quit
            break
        elif chr(key) in GESTURE_LABELS:
            label = GESTURE_LABELS[chr(key)]
            all_keypoints.append(combined_keypoints)
            all_labels.append(label)
            print(f"Recorded {label} gesture with {len(keypoints)} hands")

all_keypoints = np.array(all_keypoints)
all_labels = np.array(all_labels)

np.save(r"data\new_hand_gesture_data.npy", {'keypoints': all_keypoints, 'labels': all_labels})

cap.release()
cv2.destroyAllWindows()
