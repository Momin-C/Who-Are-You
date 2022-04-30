import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create() 
recognizer.read("recognizers/trainer.yml")

labels = {}
with open("recognizers/labels.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_colour = frame[y:y+h,x:x+w]
        
        #Facial recognizer
        id_, conf = recognizer.predict(roi_gray)
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        color = (255,255,255)
        stroke = 2
        cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        
        print(labels[id_])

        img_item = "image.png"
        cv2.imwrite(img_item, roi_colour)

        color = (255, 0, 0)
        stroke = 2
        end_cordX = x + w
        end_cordY = y + h
        cv2.rectangle(frame, (x,y),(end_cordX, end_cordY), color, stroke)
        
        '''
        #Eye and nose recognizer
        subitem = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in subitem:
            cv2.rectangle(roi_colour,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        '''

    cv2.imshow('frame',frame)

    if (cv2.waitKey(20) and 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()