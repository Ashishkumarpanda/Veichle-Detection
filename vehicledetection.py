import cv2
import matplotlib.pyplot as plt
import numpy as np
classifier=cv2.CascadeClassifier("cars.xml")
cap=cv2.VideoCapture("video1.avi")
while cap.isOpened():
    _,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cars=classifier.detectMultiScale(gray,1.4,2)

    for (x,y,w,h) in cars:
         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
         cv2.putText(img,"cars",(x,y),cv2.FONT_ITALIC,1.5,(0,0,255),2)
         cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
