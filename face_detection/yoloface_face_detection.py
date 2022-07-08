# import all libraries required to detect faces using yoloface
from yoloface import face_analysis
import cv2
import numpy as np

face = face_analysis()
#  Auto Download a large weight files from Google Drive.
#  only first time.
#  Automatically  create folder .yoloface on cwd.
# refer docs (https://pypi.org/project/yoloface/#description)


def yoloface_detection(frame):
    '''returns a list of faces detected in the frame(cropped images)
    frame-  numpy array of a image
    '''
    img, box, conf = face.face_detection(
        frame_arr=frame, frame_status=True, model='full')
    # box is a list of bounding boxes of faces
    # box[i]=[x,y,w,h]
    cropped_images = []
    faces_rect = np.array([np.array(xi) for xi in box])
    for (x, y, w, h) in faces_rect:  # to detect and crop multiple faces in each frame
        cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 255, 0), thickness=2)
        cv2.imwrite('yolo.jpg', frame)
        crop_img = frame[y:y+w, x:x+h]
        if len(crop_img) != 0:
            cropped_images.append(crop_img)
    return cropped_images
