# Emotion classification using Landmarks

This code base was created as part of a Masters level college module.

## Goals

* Use dlib and OpenCv to implement happy or sad classification.
* Use facial landmarks to generate input features to a suitable Model.
* Implement Model on a live video feed.

## Prerequisites
* Clone this repo
* Create virtual env
* pip install requirements.txt
```commandline
python3.7 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Dataset

At a high level images faces with labeled emotions is used to generate feature
vectors to train a Model.
* Dataset used: [Here](https://github.com/muxspace/facial_expressions)
* The above dataset consists of face images with associated emotions

## Image Preprocessing and generate raw dataset

* Face Alignment
```python
from imutils.face_utils.facealigner import FaceAligner

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
``` 
* Extract Features  
```python
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
```

![Landmarks](landmark_points.jpg)

## Results
    Prediction score: 0.8676470588235294

```python
def prepare_data():
    """
    Prepare training and test data
    """
    sad_data = pd.read_csv('dataset/sad_data.csv')
    sad_training, sad_test = train_test_split(sad_data, random_state=7)
    # Due to the low number of sad images only 150 happy images are picked
    happy_data = pd.read_csv('dataset/happy_data.csv')[:150]
    happy_training, happy_test = train_test_split(happy_data, random_state=7)
    # neutral_data = pd.read_csv('dataset/neutral_data.csv')
    # neutral_training, neutral_test = train_test_split(neutral_data, random_state=7)

    training_data = pd.concat([sad_training, happy_training])  #, neutral_training])
    testing_data = pd.concat([sad_test, happy_test])  #, neutral_test])

    return training_data, testing_data


train_data, test_data = prepare_data()
train_x, train_y = train_data.drop(columns=['emotion']), train_data['emotion']
test_x, test_y = test_data.drop(columns=['emotion']), test_data['emotion']

model = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=600, random_state=7)
model.fit(train_x, train_y)

with open('model.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

prediction = model.predict(test_x)
score = accuracy_score(test_y, prediction)
print(f'Prediction score: {score}')
print(test_y.value_counts())
cm = confusion_matrix(test_y, prediction, normalize=None)
plot = ConfusionMatrixDisplay(cm, ['happy', 'sad'])
plot.plot(xticks_rotation=90)
result_string = f'{model}: Score: {round(score, 3)}'
plot.ax_.set_title(result_string)
plt.subplots_adjust(bottom=0.25, right=0.80, top=0.75)
plt.show()
```

![Confusion matrix](cm.png)

## Using the model on live feed

The model can be used by executing the following script.
```commandline
python live_image_classify.py
```

Press 'q' to stop the script.

```python
import dlib
import cv2
from imutils import resize
from image_processing.pre_process import extract_features
import pickle

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
with open('model.pickle', 'rb') as handle:
    model = pickle.load(handle)

video = cv2.VideoCapture(0)

while True:
    ret, image = video.read()
    image = resize(image, 500)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(image_grey, 1)
    if faces:
        for face in faces:
            landmarks = predictor(image_grey, face)
            landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
            features = extract_features(landmarks, 'unknown')[:-1]

            emotion = model.predict([features])[0]
            cv2.putText(image, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

```

## Final comments
The classifier performs well on the test set scoring 86%.  
When using the model on a live video capture it can easily detect a happy face.  
It has some trouble identifying sad faces, you need to really frown in order for a sad classification.
Additional training of the model is needed with more training data. Only roughly 300 images were used.
