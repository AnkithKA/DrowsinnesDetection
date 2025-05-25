import cv2
import dlib
from scipy.spatial import distance
import pygame

# EAR calculation function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Mouth Aspect Ratio (MAR) calculation for yawn detection
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Load face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat')  # Ensure this file is downloaded

# Eye landmarks
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# Mouth landmarks (for yawn detection)
(mStart, mEnd) = (48, 68)

# EAR threshold and frame count
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
COUNTER = 0

# Yawn detection threshold (Mouth Aspect Ratio)
MAR_THRESHOLD = 0.50
MAR_CONSEC_FRAMES = 20
YAWN_COUNTER = 0

# Initialize pygame for sound
pygame.mixer.init()

# Load the buzzer sound (make sure you have a 'buzzer.wav' file in the same directory)
buzzer_sound = pygame.mixer.Sound('alarm.wav')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    for face in faces:
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        
        # Get eye landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Calculate EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR) / 2.0
        
        # Yawn detection using mouth landmarks
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        
        # Check for drowsiness (EAR)
        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= EAR_CONSEC_FRAMES:
                buzzer_sound.play()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            COUNTER = 0
        
        # Check for yawning (MAR)
        if mar > MAR_THRESHOLD:
            YAWN_COUNTER += 1
            if YAWN_COUNTER >= MAR_CONSEC_FRAMES:
                buzzer_sound.play()
                cv2.putText(frame, "YAWNING ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            YAWN_COUNTER = 0
    
    # Display the frame
    cv2.imshow("Driver Drowsiness and Yawn Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
