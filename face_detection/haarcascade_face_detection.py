import cv2

haarCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def haarcascade_face_detection(frame):
    '''returns a list of faces detected in the frame(cropped images)
    frame-numpy array of a image
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rect = haarCascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=9)
    cropped_images = []
    for (x, y, w, h) in faces_rect:  # to detect and crop multiple faces in each frame
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        crop_img = frame[y:y+h, x:x+w]
        if len(crop_img) != 0:
            cropped_images.append(crop_img)
    return cropped_images
