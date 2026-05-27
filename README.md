# Workshop 5 - License-plate-Detection
## Name: Ashwath M
## Register number: 212223230023

## Aim
To implement a license plate detection system using OpenCV and a Haar Cascade Classifier (haarcascade_russian_plate_number.xml) to automatically locate vehicle license plates in images.

## Software Requirements
Python 3.x

OpenCV (cv2)

NumPy

Sample vehicle images

haarcascade_russian_plate_number.xml
## Algorithm Steps
Step 1:
Read the input image and convert it to grayscale for easier processing.

Step 2:
Load the Haar Cascade Classifier for license plate detection (haarcascade_russian_plate_number.xml).

Step 3:
Apply detectMultiScale() to locate potential license plate regions in the image.

Step 4:
Draw rectangles around the detected license plate regions to highlight them.

Step 5:
Display or save the output image showing detected license plates.

## Program:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("car_plate.jpg")

def display(img):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)

display(img)

plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

def detect_plate(img):
    plate_img = img.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.3, minNeighbors = 3)

    for (x,y,w,h) in plate_rects:
        cv2.rectangle(plate_img,(x,y),(x+w,y+h),(0,0,255), 4)
    return plate_img

result = detect_plate(img)

display(result)

def detect_and_blur_plate(img):
    plate_img = img.copy()
    roi = img.copy()

    plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.3, minNeighbors=3)

    for (x,y,w,h) in plate_rects:

        roi = roi[y:y+h,x:x+w]
        blurred_roi = cv2.medianBlur(roi,7)

        plate_img[y:y+h,x:x++w] = blurred_roi
    return plate_img


result = detect_and_blur_plate(img)

display(result)


```

## Output:
<img width="972" height="565" alt="image" src="https://github.com/user-attachments/assets/c41af4bb-2d28-40ef-969b-4a5ecb0ed918" />
<img width="971" height="552" alt="image" src="https://github.com/user-attachments/assets/23a4d8ef-cefb-4c7b-b403-fb8aa664a2c7" />
<img width="976" height="551" alt="image" src="https://github.com/user-attachments/assets/717b7410-9d6f-44f9-98bf-6daf4eab023e" />


## Result
The system successfully detects and marks the license plate region(s) in the given vehicle image using a green rectangle.
