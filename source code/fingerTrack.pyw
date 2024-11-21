import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Drawing variables
drawing = False
prev_x, prev_y = None, None
canvas = None

# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if canvas is None:
        canvas = np.ones_like(frame) * 255  # Create a white canvas
    
    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of the index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Start drawing if the index finger is detected
            if not drawing:
                prev_x, prev_y = x, y
                drawing = True
            else:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)
                prev_x, prev_y = x, y

    # Display hand tracking in one window
    cv2.imshow('Hand Tracker', frame)
    
    # Display drawing in another window with a white background
    cv2.imshow('Air Drawing', canvas)
    
    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
