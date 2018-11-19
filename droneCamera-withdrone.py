import numpy as np
import cv2
import time

# drone imports
# from pyardrone import ARDrone



# Create face cascade
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

# Capture camera
cap = cv2.VideoCapture(0) # device default camera
# cap = cv2.VideoCapture("tcp://192.168.1.1:5555") # drone camera

# define color constants for quick reference
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize drone
# drone = ARDrone()

# Wait for NavData and take off
# drone.navdata_ready.wait()
# drone.takeoff()


# Main loop
while(True):
    # capture frame-by-frame
    ret, frame = cap.read()

    # cascade requires grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect face in grayscale image using cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # define some reference variables
    allowance = 45  # how far from perfect center is acceptable
    frameCenterX = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2) # exact center of frame X
    frameCenterY = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2) # exact center of frame Y

    # print(len(faces))

    # Loop if face is detected
    for(x, y, w, h) in faces:
        # print(x, y, w, h)               # print the coords of the face

        # region of interest - isolates just the detected face
        roi_gray = gray[y:y+h, x:x+w] # TODO: swap x and y for consistency?

        faceCenterX = int(x+(w/2))  # exact center of face X
        faceCenterY = int(y+(h/2))  # exact center of face Y

        # save last seen face
        # cv2.imwrite("last_seen", roi_gray)

        # Compare center of face to central target zone
        # default outline to green, change to red if off center
        # Note: these directions are from the drone's perspective
        color = GREEN
        if faceCenterX < frameCenterX - allowance:
            cv2.putText(frame,'Move Left',(50, 50), font, 1, WHITE, 1, cv2.LINE_AA)
            color = RED
        if faceCenterX > frameCenterX + allowance:
            cv2.putText(frame,'Move Right',(50, 50), font, 1, WHITE, 1, cv2.LINE_AA)
            color = RED

        if faceCenterY < frameCenterY - allowance:
            cv2.putText(frame,'Move Up',(50, 100), font, 1, WHITE, 1, cv2.LINE_AA)
            color = RED
        if faceCenterY > frameCenterY + allowance:
            cv2.putText(frame,'Move Down',(50, 100), font, 1, WHITE, 1, cv2.LINE_AA)
            color = RED

        if h > int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.6):
            cv2.putText(frame,'Move Back',(50, 150), font, 1, WHITE, 1, cv2.LINE_AA)
            color = RED
        if h < int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.3):
            cv2.putText(frame,'Move Forward',(50, 150), font, 1, WHITE, 1, cv2.LINE_AA)
            color = RED

        # whole face zone
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # face central dot
        cv2.rectangle(frame, (faceCenterX, faceCenterY), (faceCenterX, faceCenterY), WHITE, 2)

    # Draw zones
    # center target zone
    cv2.rectangle(frame, (frameCenterX - allowance, frameCenterY - allowance), (frameCenterX + allowance, frameCenterY + allowance), WHITE, 1)

    # Write drone data
    # cv2.putText(frame, str(drone.state), (50, 300), font, 0.5, WHITE, 1, cv2.LINE_AA)
    # cv2.putText(frame, str(drone.navdata.ctrl_state), (50, 350), font, 0.5, WHITE, 1, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Camera', frame)

    # Check for quit input
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# End program by releasing capture and closing windows
# drone.land()
cap.release()
cv2.destroyAllWindows()
