

from djitellopy import Tello
import cv2
import numpy as np
import imutils
import time
import threading

font = cv2.FONT_HERSHEY_COMPLEX

tello = Tello()
tello.connect()
tello.streamon()

cap = tello.get_video_capture();
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while True:
    img_rgb = (cap.read())[1]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)



    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3
    )

    if (len(faces) > 0):
        print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_rgb, str(w), (x, y), font, 1, (0, 255, 0))

            #if (w < 200 and airborne):
            #    print("move to face")
            #    tello.move_forward(10)
            #    time.sleep(1)

    cv2.imshow('thresh', img_rgb)
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break



cv2.destroyAllWindows()
