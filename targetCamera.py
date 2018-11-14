import numpy as np
import cv2
import time

# create face cascade
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

# capture camera
cap = cv2.VideoCapture(0) # device default camera
# cap = cv2.VideoCapture("tcp://192.168.1.1:5555") # drone camera

# set to 480p
cap.set(3, 640)
cap.set(4, 480)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    allowance = 30  # how far from perfect center is acceptable

    for(x, y, w, h) in faces:
        # print(x, y, w, h)               # print the coords of the face
        roi_gray = gray[y:y+h, x:x+w]   # y=origin+height, x=origin+width

        centerX = int(x+(w/2))  # X coordinate for center of the face
        centerY = int(y+(h/2))  # Y coordinate for center of the face


        # save last seen face
        img_item = "my_image.png"       # name of the image to be saved
        cv2.imwrite(img_item, roi_gray) # saves region of interest as an image

        # draw the rect
        color = (255, 0, 0) #BGR
        stroke = 2
        width = x+w
        height = y+h
        # (image to draw on, (top left coordinates), (bottom right coordinates), line color, line thickness)
        cv2.rectangle(frame, (x, y), (width, height), color, stroke)    # face outline
        cv2.rectangle(frame, (centerX - allowance, centerY - allowance), (centerX + allowance, centerY + allowance), (0,255,0), stroke)    # allowance outline

        # Finding the center
        if centerX < 320 + allowance:
            print("move left")
        if centerX > 320 + allowance:
            print("move right")
        if centerY < 240 + allowance:
            print("move down")
        if centerY > 240 + allowance:
            print("move up")

    # draw center target, assuming 480p resolution
    cv2.rectangle(frame, (320 - allowance, 240 - allowance), (320 + allowance, 240 + allowance), (0, 0, 255), 1)    # allowed target zone

    # Display the resulting frame
    cv2.imshow('frame', frame)
    # cv2.imshow('gray', gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()