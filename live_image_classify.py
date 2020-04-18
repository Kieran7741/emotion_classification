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




