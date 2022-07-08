from yoloface_face_detection import yoloface_detection
from mtcnn_face_detection import mtcnn_face_detection
from haarcascade_face_detection import haarcascade_face_detection
import cv2
import copy
import os
from pathlib import Path


def readvideo(file, start_frame=100, end_frame=400):
    '''read video and return list of cropped faces for all the models (yolo_face,haarcascade,mtcnn)
    '''
    yolo_cropped_images = []
    haar_cropped_images = []
    mtcnn_cropped_images = []

    cap = cv2.VideoCapture(file)
    count = 0
    while True:
        is_frame_available, frame = cap.read()
        count += 1
        if is_frame_available:
            new_frame1 = copy.deepcopy(frame)
            new_frame2 = copy.deepcopy(frame)
            new_frame3 = copy.deepcopy(frame)
            if count < start_frame:
                continue

            yolo_cropped = yoloface_detection(new_frame1)
            haar_cropped = mtcnn_face_detection(new_frame2)
            mtcnn_cropped = haarcascade_face_detection(new_frame3)

            yolo_cropped_images.extend(yolo_cropped)
            haar_cropped_images.extend(haar_cropped)
            mtcnn_cropped_images.extend(mtcnn_cropped)

            print(count)
            if count == end_frame:
                break
        else:
            break

    cap.release()
    return yolo_cropped_images, haar_cropped_images, mtcnn_cropped_images


def get_crop_images(crop_images, path, model_name):
    """ crop_images is a list of images(numpy array of list)
        path is the path to the folder where the images are to be saved
        model name is the name of the model used to detect faces(inside path model_name folder is created and images are saved)    
    """
    print(os.path.join(path,  model_name))
    os.mkdir(os.path.join(path, model_name))
    for i in range(len(crop_images)):
        image = crop_images[i]
        file_path = os.path.join(path, model_name, str(i))
        print(image)
        cv2.imwrite(file_path+".jpg", image)


def get_files(folder_path):
    '''return list of files inside a folder'''
    files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        files.append(file_path)
    return files


def get_all_faces():
    curent_working_directory = os.getcwd()
    # print(curent_working_directory)
    # make sure you are running (python3 main.py) inside the face_detection folder

    input_dataset_directory = os.path.join(
        curent_working_directory, 'data', 'videos')
    # videos should be inside data folder

    ouput_dataset_directory = os.path.join(
        curent_working_directory, "data", "cropped_images")
    os.mkdir(ouput_dataset_directory)
    # output dataset directory should be inside data folder with the name cropped_images

    files = get_files(input_dataset_directory)
    for file in files:
        yolo_croppes_faces, haar_case_cade_faces, mtcnn_cropped_faces = readvideo(
            file, 50, 53)
        # for start and end frame initiallly set to 100 and 102(just to test everything is working fine)
        # then set start and end frame to 100 and 400 to create dataset
        # set start frame to 1 and end frame to -1 to create dataset for all frames in video(this takes more time)

        folder_name = Path(file).stem
        folder_name = os.path.join(ouput_dataset_directory, folder_name)
        os.mkdir(folder_name)
        # this creates a folder for each video with name same as folder name inside ouput_dataset_directory

        get_crop_images(yolo_croppes_faces, folder_name, "yolo")
        get_crop_images(haar_case_cade_faces, folder_name, "haar")
        get_crop_images(mtcnn_cropped_faces, folder_name, "mtcnn")


def main():
    get_all_faces()


if __name__ == '__main__':
    main()
