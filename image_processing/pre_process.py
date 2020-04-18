import cv2
import dlib
import imutils
from imutils.face_utils.facealigner import FaceAligner, FACIAL_LANDMARKS_68_IDXS
import pandas as pd
import os
from math import sqrt


def area_of_polygon(points):
    """
    Get area of a polygon.
    Modified version of: https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon
    :param points:
    :return: Area in pixels
    """
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in zip(points, points[1:] + [points[0]])))


def line_length(p1, p2):
    """
    Calculate length between two points
    """
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def slope_of_line(p1, p2):
    """
    Calculate slope of line
    """
    return (p1[1] - p2[1]) / (p1[0] - p2[0])


def get_list_of_images(legend, image_path, emotion='happiness'):
    """Fetch a list of images for the provided emotion"""

    csv_legend = pd.read_csv(legend)

    return [os.path.join(image_path, image) for image in csv_legend[csv_legend['emotion'] == emotion]['image']]


def align_face(image_path, save_image=True):
    """
    Standardize face orientation of face.
    :param image_path: Path to image
    :param save_image: Overwrite existing image
    """
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = detector(image, 1)
    if faces:
        for face in faces:
            aligned_image = FaceAligner(predictor, desiredFaceWidth=300).align(image, image, face)
            if save_image:
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
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
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


def extract_features(landmarks, emotion):
    """
    Extract a list of features from the provided face landmarks
        Features:
            0: eye_area
            1: inner_mouth_area -> Larger indicates open mouth potentially due to smiling
            2: mouth_area -> Larger indicates larger mouth due to smiling
            3: lip_slope -> slope of lips
            3: lip_to_eye_distance -> Shorter indicated smiling

    :param landmarks: Face landmarks
    :param emotion: Associated emotion with landmarks. Appended to feature list
    :return: Feature vector
    """

    right_eye_slice = FACIAL_LANDMARKS_68_IDXS['right_eye']
    left_eye_slice = FACIAL_LANDMARKS_68_IDXS['left_eye']
    inner_mouth_slice = FACIAL_LANDMARKS_68_IDXS['inner_mouth']
    outer_mouth_slice = (48, 60)

    # Distance to normalize all measurements
    nose_to_inner_eye_distance = line_length(landmarks[33], landmarks[39])

    mouth_area = area_of_polygon(landmarks[outer_mouth_slice[0]:outer_mouth_slice[1]]) / nose_to_inner_eye_distance
    inner_mouth_area = (area_of_polygon(landmarks[inner_mouth_slice[0]:inner_mouth_slice[1]]) or 1) / nose_to_inner_eye_distance
    right_eye_area = area_of_polygon(landmarks[right_eye_slice[0]:right_eye_slice[1]])
    left_eye_area = area_of_polygon(landmarks[left_eye_slice[0]:left_eye_slice[1]])
    average_eye_area = ((right_eye_area + left_eye_area) / 2) / nose_to_inner_eye_distance

    lip_slope = slope_of_line(landmarks[48], landmarks[50])

    lip_distance_to_eye = line_length(landmarks[36], landmarks[48])

    return [average_eye_area, inner_mouth_area, mouth_area,
            lip_slope, lip_distance_to_eye, emotion]

