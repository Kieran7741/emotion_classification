# Emotion classification using Landmarks

This code base was created as part of a Masters level college module.

## Goals

* Use dlib and OpenCv to implement happy or sad classification.
* Use facial landmarks to generate input features to a suitable Model.
* Implement Model on a live video feed.

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

![Landmarks](landmark_points.jpg)

## Results
    Prediction score: 0.8676470588235294

![Confusion matrix](cm.png)

