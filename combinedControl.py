from skimage import io      # Importing the io module from the scikit-image library.
import threading            # Importing the threading module to multi-thread the code.
import cv2                  # Importing opencv-python to work on video face recognition.
from pyardrone import ARDrone


# Initializing the global variables that will be used throughout the script.
running = True                  # running global set to true.
frame = 0                       # frame global set to 0.
# Creating a VideoCapture object with the default camera passed as its argument.
cam = cv2.VideoCapture('tcp://192.168.1.1:5555')       # 'tcp://192.168.1.1:5555'
# Creating a CascadeClassifier object to identify frontal faces.
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# Define color constants for quick reference
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
# Initialize drone
drone = ARDrone()


def verify_image(img_file):
    """
    The verify_image function verifies whether the given image is complete or not.
    :param img_file: the image to verify.
    :return: True if the image is complete or False otherwise.
    """
    try:
        image = io.imread(img_file)
    except:
        return False
    return True


def make_720p():
    """
    The make_720p function sets the resolution of the screen to 720p.
    """
    cam.set(3, 1280)
    cam.set(4, 720)


def access_camera():
    """
    The access_camera function accesses the camera to get the given frame.
    """
    global running      # Declaring the running global variable to be able to change it.
    global frame        # Declaring the frame global variable to be able to change it.
    # This while loop gets a frame per iteration of the loop.
    while True:
        running, frame = cam.read()


def save_and_show():
    """
    The save_and_show function saves the access_camera function's given frame, shows it and
    recognizes the faces within the frame.
    """
    # This while loop saves the given frame, takes it back to show it up in the screen and executes the
    # face recognition operations on the given frame.
    while True:
        if running:     # If the frame was successfully read.
            cv2.imwrite('image/now.jpg', frame)         # Saving frame into the system as an image.
            img = cv2.imread('image/now.jpg', 1)        # Reading the image back from the system.

            if verify_image('image/now.jpg'):       # If the image is not corrupted.
                # Creating a copy of the imgage in grey scale.
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Creating a tuple with a list of detected faces within the image.
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

                # define some reference variables
                allowance = 45  # how far from perfect center is acceptable
                frameCenterX = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2) # exact center of frame X
                frameCenterY = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2) # exact center of frame Y
                power = 0.3 # how much power to give drone motors

                # This for loop iterates thought the tuple to locate the detected faces within the image.
                for (x, y, w, h) in faces:
                    # Detecting one face at the time by setting each detected face equals to the first face coordinates.
                    (x, y, w, h) = faces[0]
                    # Printing the coordinates given by the face.
                    # print(x, y, w, h)

                    # Reset directional power values
                    pwr_forward = 0
                    pwr_back = 0
                    pwr_left = 0
                    pwr_right = 0
                    pwr_up = 0
                    pwr_down = 0

                    # Compare center of face to central target zone
                    # default outline to green, change to red if off center
                    # Note: these directions are from the drone's perspective
                    color = GREEN
                    if faceCenterX < frameCenterX - allowance:
                        cv2.putText(frame,'Move Left',(50, 50), font, 1, WHITE, 1, cv2.LINE_AA)
                        pwr_left = power
                        color = RED
                    elif faceCenterX > frameCenterX + allowance:
                        cv2.putText(frame,'Move Right',(50, 50), font, 1, WHITE, 1, cv2.LINE_AA)
                        pwr_right = power
                        color = RED

                    if faceCenterY < frameCenterY - allowance:
                        cv2.putText(frame,'Move Up',(50, 100), font, 1, WHITE, 1, cv2.LINE_AA)
                        pwr_up = power
                        color = RED
                    elif faceCenterY > frameCenterY + allowance:
                        cv2.putText(frame,'Move Down',(50, 100), font, 1, WHITE, 1, cv2.LINE_AA)
                        pwr_down = power
                        color = RED

                    if h > int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.6):
                        cv2.putText(frame,'Move Back',(50, 150), font, 1, WHITE, 1, cv2.LINE_AA)
                        pwr_back = power
                        color = RED
                    elif h < int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.3):
                        cv2.putText(frame,'Move Forward',(50, 150), font, 1, WHITE, 1, cv2.LINE_AA)
                        pwr_forward = power
                        color = RED

                    # whole face zone
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    # face central dot
                    cv2.rectangle(frame, (faceCenterX, faceCenterY), (faceCenterX, faceCenterY), WHITE, 2)
                # end for loop

                # Draw zones
                # center target zone
                cv2.rectangle(frame, (frameCenterX - allowance, frameCenterY - allowance), (frameCenterX + allowance, frameCenterY + allowance), WHITE, 1)

                # Write drone data
                cv2.putText(frame, str(drone.state), (50, 300), font, 0.5, WHITE, 1, cv2.LINE_AA)
                cv2.putText(frame, str(drone.navdata.ctrl_state), (50, 350), font, 0.5, WHITE, 1, cv2.LINE_AA)

                # Display the resulting frame
                cv2.imshow('Camera', frame)

                # Actually move the drone
                drone.move(forward=pwr_forward, backward=pwr_back, left=pwr_left, right=pwr_right, up=pwr_up, down=pwr_down, cw=0, ccw=0)
                time.sleep(1)

                # Showing the frame up on the screen.
                cv2.imshow('frame', img)
                # This if statements breaks the loop if the "q" key is pressed.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    drone.land()
                    break
    # end while loop

    cam.release()            # Releasing VideoCapture object.
    cv2.destroyAllWindows()  # Destroying all created windows.


def main():
    if __name__ == "__main__":      # If the current executing thread is the main function.

        # Wait for NavData and take off
        drone.navdata_ready.wait()
        # while not drone.state.fly_mask:
        #     drone.takeoff()

        # Creating the first thread with the access_camera function as its target.
        t1 = threading.Thread(target=access_camera)
        # Creating the second thread with the save_and_show function as its target.
        t2 = threading.Thread(target=save_and_show)

        t1.start()      # Running the first thread.
        t2.start()      # Running the second thread.

        t1.join()       # Waiting until the first thread finishes before continuing.
        t2.join()       # Waiting until the second thread finishes before continuing.


main()

# force drone land on program end
drone.land()