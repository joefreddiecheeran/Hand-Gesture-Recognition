import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test = np.load('data/X_test.npy') 
y_test = np.load('data/y_test.npy')  

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_categorical = to_categorical(y_train_encoded)

y_test_encoded = label_encoder.transform(y_test)  

model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_categorical, epochs=50, batch_size=32)

model.save('models/gesture_model.h5')
np.save('models/label_encoder.npy', label_encoder.classes_)

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)

print("Multilayer Perceptron Accuracy: ", accuracy_score(y_test_encoded, y_pred_classes))
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred_classes, target_names=label_encoder.classes_))
