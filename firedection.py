import numpy as np
import cv2
import time

fire_cascade = cv2.CascadeClassifier('fire_detection.xml')

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)
while 1:

    ret, frame = cap.read()
    cv2.imshow('imo', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(frame, 1.2, 5)
    for (x, y, w, h) in fire:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        print('Fire is detected..!')

        time.sleep(0.2)

    cv2.imshow('img', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()