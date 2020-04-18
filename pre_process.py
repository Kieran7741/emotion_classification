import cv2
import dlib
import imutils
from imutils.face_utils.facealigner import FaceAligner, FACIAL_LANDMARKS_68_IDXS
import pandas as pd
import os
import csv


def area_of_polygon(points):
    """
    Get area of a polygon.
    Modified version of: https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon
    :param points: 
    :return: Area in pixels
    """
    print(points)
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in zip(points, points[1:] + [points[0]])))


def slope_of_line(p1, p2):
    """
    Calculate slope of line
    :param p1:
    :param p2:
    :return: Slope of line
    """
    return (p1[1] - p2[1]) / (p1[0] - p2[0])


def get_list_of_images(emotion='happiness'):
    """Fetch a list of images for the provided emotion"""
    file = '//fer_2013/data/legend.csv'
    image_path = '//fer_2013/images/{0}'

    csv = pd.read_csv(file)

    return [image_path.format(image) for image in csv[csv['emotion'] == emotion]['image']]


def get_width_height_of_face(face):
    w = face.right() - face.left()
    h = face.bottom() - face.top()
    return w, h


def align_face(image_path, save_image=True):
    """
    Standardize face orientation of face and
    :param image_path:
    :param save_image:
    :return:
    """
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = detector(image, 1)
    if faces:
        for face in faces:
            aligned_image = FaceAligner(predictor, desiredFaceWidth=300).align(image, image, face)
            if save_image:
                print(image_path)

                cv2.imwrite(image_path, aligned_image)
            else:
                cv2.imshow("Aligned", aligned_image)
                cv2.waitKey(0)


def get_landmarks_from_images(image_paths, num_landmarks=68):
    """
    Get landmarks from the provided images.
    :param image_paths:
    :param num_landmarks:
    :return:
    """

    faces_landmarks = []
    for image_path in image_paths:
        if os.path.exists(image_path):
            image = imutils.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), width=300)
            faces = detector(image, 1)
            if faces:
                for face in faces:
                    landmarks = predictor(image, face)
                    faces_landmarks.append([(landmarks.part(i).x, landmarks.part(i).y) for i in range(num_landmarks)])
            else:
                print(f'No face found in image: {image_path}')
        else:
            print(f'Image not found {image_path}')

    return faces_landmarks

# def get_face_area(landmarks):
#
# def get_average_eye_area(landmarks):
#     """
#     Get the average area of both eyes
#     :param landmarks:
#     :return:
#     """


def extract_features(landmarks, emotion):
    """
    Extract a list of features from the provided face landmarks
        Features:
            0: eye_area (average of both eyes)
            1: inner_mouth_area -> teeth
            2: eye_inner_mouth_ratio -> smaller eyes due to squinting caused by smiling
            3: lip_area
            4: lip_angle -> flatter slope indicates smiling

    :param landmarks:
    :param emotion:
    :return:
    """

    right_eye_slice = FACIAL_LANDMARKS_68_IDXS['right_eye']
    left_eye_slice = FACIAL_LANDMARKS_68_IDXS['left_eye']
    inner_mouth_slice = FACIAL_LANDMARKS_68_IDXS['inner_mouth']
    outer_mouth_slice = (48, 61)
    mouth_area = area_of_polygon(landmarks[outer_mouth_slice[0]:outer_mouth_slice[1]])
    inner_mouth_area = area_of_polygon(landmarks[inner_mouth_slice[0]:inner_mouth_slice[1]]) or 1
    right_eye_area = area_of_polygon(landmarks[right_eye_slice[0]:right_eye_slice[1]])
    left_eye_area = area_of_polygon(landmarks[left_eye_slice[0]:left_eye_slice[1]])
    average_eye_area = (right_eye_area + left_eye_area) / 2
    lip_slope = slope_of_line(landmarks[48], landmarks[51])
    return [average_eye_area, inner_mouth_area, mouth_area, average_eye_area/inner_mouth_area,
            lip_slope, emotion]



