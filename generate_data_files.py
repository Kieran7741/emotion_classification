import os
from image_processing.pre_process import get_landmarks_from_images, get_list_of_images, extract_features
import dlib
import csv


if __name__ == '__main__':
    # Load images

    legend_csv = './fer_2013/data/legend.csv'
    base_image_path = './fer_2013/images'

    sad_images, happy_images, neutral_images = (get_list_of_images(legend_csv, base_image_path, 'sadness'),
                                                get_list_of_images(legend_csv, base_image_path, 'happiness')[:500],
                                                get_list_of_images(legend_csv, base_image_path, 'neutral')[:500])
    print(f'Sad images: {len(sad_images)}; Happy images: {len(happy_images)}, '
          f'Neutral images: {len(neutral_images)}')

    # align images
    # sad_images + happy_images[:1000] +
    # for i, image_path in enumerate(neutral_images[:1000]):
    #     print(f'Image: {i}/{len(sad_images)+ len(happy_images) + len(neutral_images)}')
    #     align_face(image_path)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    print('Extracting landmarks...')
    sad_image_landmarks = get_landmarks_from_images(sad_images)
    happy_image_landmarks = get_landmarks_from_images(happy_images)
    neutral_image_landmarks = get_landmarks_from_images(neutral_images)

    emotion_map = {'sad': sad_image_landmarks,
                   'happy': happy_image_landmarks,
                   'neutral': neutral_image_landmarks}

    if not os.path.exists('./dataset'):
        os.mkdir('./dataset')

    for emotion in emotion_map.keys():
        with open(f'dataset/{emotion}_data.csv', 'w', newline='') as f:
            wr = csv.writer(f)
            f.write('eye_area,inner_mouth_area,mouth_area,lip_slope,lip_distance_to_eye,emotion\n')
            for landmark in emotion_map[emotion]:
                data = extract_features(landmark, emotion)
                wr.writerow(data)