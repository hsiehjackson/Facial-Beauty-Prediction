import cv2
import sys
from PIL import Image, ImageDraw, ImageFont
from score import *
import warnings
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--cascPath', type=str, default="xml/haarcascade_frontalface_default.xml")
parser.add_argument('--model',type=str, default='model/squeeze-0.218914.pkl')
args = parser.parse_args()

faceCascade = cv2.CascadeClassifier(args.cascPath)
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 120)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(frame.shape)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)        
        img = Image.fromarray(frame[y:y+h,x:x+w,:])
        #cv2.imshow('image',frame[y:y+h,x:x+w,:])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        mean, std = score(img, args.model)
        mean = round(mean, 2)
        std = round(std,2)
        #print('beauty score: {:.2f}%{:.2f}'.format(mean,std))
        cv2.putText(frame,str(mean)+'%'+str(std),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0))
        
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()