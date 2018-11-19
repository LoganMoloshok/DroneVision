from skimage import io
import cv2


def make_720p():
    """The make_720p function sets the resolution of the screen to 720p."""
    cam.set(3, 1280)
    cam.set(4, 720)


def verify_image(img_file):
    try:
        image = io.imread(img_file)
    except:
        return False
    return True


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cam = cv2.VideoCapture(0)  # 'tcp://192.168.1.1:5555'

# make_720p()

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

        img = cv2.imread('image/now.jpg', 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:
            (x, y, w, h) = faces[0]
            print(x, y, w, h)

            roi_gray = gray[y: y + h, x: x + w]
            roi_color = img[y: y + h, x: x + w]

            img_item = "my-image.png"
            cv2.imwrite(img_item, roi_gray)

            color = (255, 0, 0)
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), color, stroke)

            center_color = (0, 0, 255)
            center_stroke = 1
            center_cord_x = 550
            center_cord_y = 225
            center_w = 225
            center_h = 225
            center_end_cord_x = center_cord_x + 225
            center_end_cord_y = center_cord_y + 225
            cv2.rectangle(img, (center_cord_x, center_cord_y), (center_end_cord_x, center_end_cord_y),
                          center_color, center_stroke)

            if x > center_cord_x:
                print('Move right')

            if x < center_cord_x:
                print('Move left')

            if y > center_cord_y:
                print('Move up')

            if y < center_cord_y:
                print('Move down')

            if w < center_w or h < center_h:
                print('Get closer')

            if w > center_w or h > center_h:
                print('Get away')

        cv2.imshow('frame', img)

        # cv2.imshow('Frame', frame)

    else:
        # error reading frame
        print('error reading video feed')

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
