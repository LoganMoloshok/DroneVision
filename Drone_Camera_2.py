import cv2
from skimage import io


def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True


def make_1080p():
    """The make_1080 function sets the resolution of the screen to 1080p."""
    # cap.set(3, 1920)
    # cap.set(4, 1080)


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

# cap = cv2.VideoCapture(0)

make_1080p()

while True:

    # ret, frame = cap.read()
    img = cv2.imread('image/now.jpg', 1)

    if verify_image('image/now.jpg'):

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

        cv2.imshow('frame', img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

# cap.release()
cv2.destroyAllWindows()
