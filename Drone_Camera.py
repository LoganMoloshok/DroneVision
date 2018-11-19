from skimage import io
import cv2


def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True


cam = cv2.VideoCapture(0)        # 'tcp://192.168.1.1:5555'
running = True
while running:
    # get current frame of video
    running, frame = cam.read()

    if running:
        # cv2.imshow('frame', frame)
        cv2.imwrite('image/now.jpg', frame)

        if verify_image('image/now.jpg'):
            print('good')
        else:
            print('bad')
            cv2.imwrite('image/now.jpg', frame)

        cv2.imshow('Frame', frame)

    else:
        # error reading frame
        print('error reading video feed')

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
