import numpy as np
import cv2
import mediapipe as mp
import joblib  
from sklearn.preprocessing import LabelEncoder

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

rf_model = joblib.load('models/random_forest_model.joblib')

label_encoder_classes = np.load('models/label_encoder.npy', allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

cap = cv2.VideoCapture(0)

def capture_keypoints():
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:
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
                    combined_keypoints = np.concatenate((keypoints[0], np.zeros(21 * 3)))  

                return combined_keypoints.reshape(1, -1), image  

            cv2.imshow("Hand Gesture Classifier", image)
            if cv2.waitKey(10) & 0xFF == ord('0'):
                break

while True:
    keypoints_flattened, frame = capture_keypoints()
    
    prediction = rf_model.predict(keypoints_flattened)

    predicted_gesture = label_encoder.inverse_transform([prediction[0]])[0]

    cv2.putText(frame, f'Predicted Gesture: {predicted_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Classifier", frame)
    if cv2.waitKey(10) & 0xFF == ord('0'):
        break

cap.release()
cv2.destroyAllWindows()
