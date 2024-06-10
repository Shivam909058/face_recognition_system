import dlib
import cv2
import os
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def align_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) == 0:
        return None

    rect = rects[0]
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    (x, y, w, h) = face_utils.rect_to_bb(rect)
    face_aligned = cv2.resize(image[y:y + h, x:x + w], (128, 128))
    return face_aligned


def preprocess_images(input_dir='../data/user1/', output_dir='../data/user1_aligned/'):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        image = cv2.imread(os.path.join(input_dir, filename))
        aligned_face = align_face(image)
        if aligned_face is not None:
            cv2.imwrite(os.path.join(output_dir, filename), aligned_face)


if __name__ == "__main__":
    preprocess_images()
