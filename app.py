from flask import Flask, Response, jsonify
from flask_sock import Sock
from flask_cors import CORS
import cv2
import numpy as np
import json
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib
import time

app = Flask(__name__)
CORS(app)  # Enable CORS
sock = Sock(app)

# Load the trained model, scaler, and label binarizer
model = load_model('gesture_recognition_model.h5')  # Ensure this path is correct
scaler = joblib.load('scaler.pkl')  # Ensure this path is correct
label_binarizer = joblib.load('label_binarizer.pkl')  # Ensure this path is correct

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Global variable to control the webcam stream and current gesture
streaming = False
current_gesture = "No gesture detected"
displayed_gesture = "No gesture detected"  # This will hold the gesture for display purposes

def recognize_gesture(image):
    """Recognize the gesture from the image."""
    global current_gesture, displayed_gesture
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = image.shape
            hand_landmark_array = np.array([[point.x * w, point.y * h] for point in hand_landmarks.landmark]).flatten()
            hand_landmark_array = hand_landmark_array.reshape(1, -1)
            hand_landmark_array = scaler.transform(hand_landmark_array)

            predictions = model.predict(hand_landmark_array)
            predicted_label = label_binarizer.inverse_transform(predictions)[0]

            # Convert to Python int if it's a numpy type
            if isinstance(predicted_label, (np.int64, np.float64)):
                predicted_label = int(predicted_label)

            # Handle space gesture: display "Space" but treat it as a blank space internally
            if predicted_label == 'space':  # Adjust this if your 'space' label is different
                current_gesture = " "  # Internally treat it as a space character
                displayed_gesture = "Space"  # For display purposes, show "Space"
            else:
                current_gesture = predicted_label  # Update global gesture
                displayed_gesture = predicted_label  # Display the actual gesture

            return predicted_label
    else:
        current_gesture = "No gesture detected"  # Set to "No gesture detected" when no hands are found
        displayed_gesture = "No gesture detected"  # Same for display
    return None

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    """Generate frames from the webcam."""
    global streaming
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return  # Exit if the camera cannot be opened

    while streaming:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame from webcam.")
            break
        else:
            predicted_gesture = recognize_gesture(frame)

            # Display the predicted gesture on the frame
            cv2.putText(frame, f'Sign: {displayed_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/start', methods=['POST'])
def start_streaming():
    """Start video streaming."""
    global streaming
    streaming = True
    print("Streaming started.")
    return jsonify({"status": "Streaming started"}), 200

@app.route('/stop', methods=['POST'])
def stop_streaming():
    """Stop video streaming."""
    global streaming
    streaming = False
    print("Streaming stopped.")
    return jsonify({"status": "Streaming stopped"}), 200

@sock.route('/ws')
def websocket(ws):
    """WebSocket for gesture recognition."""
    global current_gesture
    while streaming:
        # Send the current gesture to the frontend as a string
        ws.send(json.dumps({'gesture': str(displayed_gesture)}))  # Send the displayed gesture to the frontend
        time.sleep(1)  # Adjust the frequency as necessary

if __name__ == '__main__':
    app.run(debug=True, port=5000)
