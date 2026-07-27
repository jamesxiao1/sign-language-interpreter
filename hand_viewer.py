import cv2 
import mediapipe as mp 
import sys 
import csv

mp_hands =mp.solutions.hands # for tracking hands 
mp_draw =mp.solutions.drawing_utils # draws the skeleton on hands 

hands =mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0, # lighter faster model, 1 is more accurate but slower
    min_detection_confidence=0.6, # min confidence to identify hand, lower=more hallucination
    min_tracking_confidence=0.5 # min confidence to keep tracking hand 
)

cap=cv2.VideoCapture(0) # open video camera 

while True: 
    ok,frame=cap.read() 
    # cap.read() grabs next frame and returns, ok (true/false if it worked), and frame (the actual image,grid of numbers)
    if not ok: 
        break

    frame =cv2.flip(frame,1) # flips frame horizontally (1)

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # mediapipe uses rgb,opencv uses bgr, need to convert 

    results=hands.process(rgb)

    if results.multi_hand_landmarks: #results.multi_hand_landmarks is either a list of hands found or none
        for hand_landmarks in results.multi_hand_landmarks: 
            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            # 3 arguments, where to draw (frame),what to draw (hand_landmarks,the 21 points),how to connect the points(HAND_CONNECTIONS)
    
    cv2.imshow("Hand Landmarks",frame) # shows frame in a window
    if cv2.waitKey(1) & 0xFF==ord("q"): # mem loc of q is FF 
        break

cap.release() # release frees webcam
cv2.destroyAllWindows() # 

