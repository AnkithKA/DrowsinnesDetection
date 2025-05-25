import cv2
import dlib
import numpy as np
from keras.models import load_model
from imutils import face_utils
from playsound import playsound
import threading

# Load Models
eye_models = [
    load_model("models/eye_cnn1.keras"),
    load_model("models/eye_cnn2.keras"),
    load_model("models/eye_cnn3.keras")
]
mouth_models = [
    load_model("models/mouth_cnn1.keras"),
    load_model("models/mouth_cnn2.keras"),
    load_model("models/mouth_cnn3.keras")
]

# Dlib setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")  # Fixed path

# To prevent overlapping buzzers
buzzer_playing = False

def play_buzzer():
    global buzzer_playing
    if not buzzer_playing:
        buzzer_playing = True
        threading.Thread(target=lambda: (playsound("alarm.wav"), set_buzzer_flag())).start()

def set_buzzer_flag():
    global buzzer_playing
    buzzer_playing = False

def crop_and_process(gray, shape, points):
    region = shape[points[0]:points[1]]
    x, y, w, h = cv2.boundingRect(np.array(region))
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (50, 50))
    roi = roi.astype("float32") / 255.0
    return roi.reshape(1, 50, 50, 1)

def ensemble_predict(models, X):
    preds = [model.predict(X, verbose=0)[0][0] for model in models]
    return np.mean(preds) > 0.8

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye_img = crop_and_process(gray, shape, (36, 42))
        right_eye_img = crop_and_process(gray, shape, (42, 48))
        mouth_img = crop_and_process(gray, shape, (48, 68))

        eye_state = ensemble_predict(eye_models, left_eye_img) or ensemble_predict(eye_models, right_eye_img)
        mouth_state = ensemble_predict(mouth_models, mouth_img)

        if eye_state or mouth_state:
            cv2.putText(frame, "DROWSINESS DETECTED!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            play_buzzer()  # Trigger buzzer
        else:
            cv2.putText(frame, "Awake", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
