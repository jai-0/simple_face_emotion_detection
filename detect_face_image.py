import cv2
from fer import FER
import matplotlib.pyplot as plt    


# Load the cascade
face_cascade = cv2.CascadeClassifier('C:\\Users\\atharva\\Documents\\codes\\facedetection-master\\facedetection-master\\haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

return_value, image = camera.read()
img=  image

detector = FER()
f=detector.top_emotion(img)
print(f)
# Read the input image
#img = cv2.imread('C:\\Users\\atharva\\Documents\\codes\\facedetection-master\\facedetection-master\\test.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
window_name = 'img'
  
# text 
text = str(f[0])  
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (00, 185) 
  
# fontScale 
fontScale = 1
   
# Red color in BGR 
color = (0, 0, 255) 
  
# Line thickness of 2 px 
thickness = 2
   
# Using cv2.putText() method 
image = cv2.putText(image, text, org, font, fontScale,  
                 color, thickness, cv2.LINE_AA, False) 
  
# Using cv2.putText() method 
#image = cv2.putText(image, text, org, font, fontScale, 
                  #color, thickness, cv2.LINE_AA, True)  
  
# Displaying the image 
cv2.imshow(window_name, image) 

# Display the output
del(camera)
#cv2.imshow('img', img)
cv2.waitKey()
