import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the trained model and the scaler
model = load_model('gesture_recognition_model.h5')
scaler = joblib.load('scaler.pkl')
label_binarizer = joblib.load('label_binarizer.pkl')

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize hand detection model
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Draw landmarks and connections on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark points
            h, w, _ = image.shape
            hand_landmark_array = np.array([[point.x * w, point.y * h] for point in hand_landmarks.landmark]).flatten()

            # Prepare the input for the model
            hand_landmark_array = hand_landmark_array.reshape(1, -1)  # Reshape for model input
            hand_landmark_array = scaler.transform(hand_landmark_array)  # Normalize

            # Predict the gesture
            predictions = model.predict(hand_landmark_array)
            predicted_label = label_binarizer.inverse_transform(predictions)[0]  # Get the label

            # Display the predicted sign on the image
            cv2.putText(image, f'Sign: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show the processed image with landmarks and predicted sign
    cv2.imshow('Hand Gesture Recognition', image)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
