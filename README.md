# Workshop 5 - License-plate-Detection
## Name: VENKATA MOHAN N
## Register number: 212224230298

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
<img width="990" height="638" alt="image" src="https://github.com/user-attachments/assets/1187c218-444c-46ef-950a-99114ffe471a" />
<img width="1028" height="651" alt="image" src="https://github.com/user-attachments/assets/18b6973e-ce81-40cf-b0cb-4184f65d265f" />

<img width="1013" height="638" alt="image" src="https://github.com/user-attachments/assets/11d9affa-b1c3-4f9e-a7b3-b64b4198dcd9" />


## Result
The system successfully detects and marks the license plate region(s) in the given vehicle image using a green rectangle.
