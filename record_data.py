import cv2 
import mediapipe as mp 
import sys 
import csv

mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils 

hands=mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

csv_file=open("data.csv","a",newline="")
writer=csv.writer(csv_file)

cap=cv2.VideoCapture(0)


while True: 
    ok,frame=cap.read()
    if not ok:
        break


    frame=cv2.flip(frame,1)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results=hands.process(rgb)

    if results.multi_hand_landmarks: 
        for hand_landmarks in results.multi_hand_landmarks: 
            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Landmarks",frame)
        key=cv2.waitKey(1) &0xFF
        if key==27: # escape
            break
        if ord("a")<=key<=ord("z"): 
            the_hand=results.multi_hand_landmarks[0] # first hand detected
            lm_list=[chr(key)]
            for lm in the_hand.landmark:
                
                lm_list.append(lm.x)
                lm_list.append(lm.y)
                lm_list.append(lm.z)

            # print(len(lm_list))
            writer.writerow(lm_list)


csv_file.close()
cap.release() 
cv2.destroyAllWindows()

