import os
import cv2
import numpy as np

haarcascade = f'{os.getcwd()}/face_detection/haarcascade_frontalface_default.xml'
print(haarcascade)
face_classifier = cv2.CascadeClassifier(haarcascade)


def detect_faces(img):
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5,0)
    
    if faces is None:
        return img
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0),2)
        
    return img
