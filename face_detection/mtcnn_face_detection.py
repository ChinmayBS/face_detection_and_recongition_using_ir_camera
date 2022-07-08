# mtcnn face detection
from mtcnn import MTCNN
mtcnn_detector = MTCNN()


def mtcnn_face_detection(frame):
    '''returns a list of faces detected in the frame(cropped images)
    frame-numpy array of a image
    '''
    features = mtcnn_detector.detect_faces(frame)
    # here features is a list of dictionaries (refer mtcnn docs for more info)
    cropped_images = []
    faces_rect = []
    for feature in features:
        faces_rect.append(feature['box'])

    for (x, y, w, h) in faces_rect:
        crop_img = frame[y:y+h, x:x+w]
        if len(crop_img) != 0:
            cropped_images.append(crop_img)
    return cropped_images
