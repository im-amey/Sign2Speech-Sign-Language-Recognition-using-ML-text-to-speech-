import cv2 # type: ignore
import mediapipe as mp # type: ignore
import time
import pyttsx3 # type: ignore
import random
from tensorflow.keras.models import load_model # type: ignore
import numpy as np # type: ignore

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
model = load_model('D:\Test_02_Sem6\Sign-Language-detection-main\Model\keras_model.h5')

label_map =  ["A","B","C","D","E","F","G","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Z"]

def predict_sign_from_image(cropped_img):
    img = cv2.resize(cropped_img, (224, 224))
    img = img.astype('float32') / 255.0 
    img = np.expand_dims(img, axis=0)  
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    return label_map[predicted_index]


# Mediapipe Hands Initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Start video capture
cap = cv2.VideoCapture(0)

# Timer
prev_time = 0
interval = 3 

last_prediction = ""

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    current_time = time.time()

    if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            landmark_array = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            x_list, y_list = zip(*landmark_array)
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

            x1, y1 = max(x_min - 20, 0), max(y_min - 20, 0)
            x2, y2 = min(x_max + 20, frame.shape[1]), min(y_max + 20, frame.shape[0])
            hand_img = frame[y1:y2, x1:x2]

            if current_time - prev_time > interval:
                prev_time = current_time
                last_prediction = predict_sign_from_image(hand_img)
                print(f"Detected: {last_prediction}")
                engine.say(last_prediction)
                engine.runAndWait()

            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

            if current_time - prev_time > interval:
                prev_time = current_time
            
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                last_prediction = predict_sign(landmarks) # type: ignore

                print(f"Detected: {last_prediction}")
                engine.say(last_prediction)
                engine.runAndWait()

    # Display subtitle
    if last_prediction:
        cv2.putText(frame, last_prediction, (275, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to end
        break

cap.release()
cv2.destroyAllWindows()
