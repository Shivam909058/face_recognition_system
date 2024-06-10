import face_recognition
import hashlib
# cf487b44a7abe82d98d23cd02daad904ce3aa0f67fc41b041c1b4cae97af705b

def get_face_hash(image):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if not face_encodings:
        return None

    face_encoding = face_encodings[0]
    face_hash = hashlib.sha256(face_encoding).hexdigest()
    return face_hash


if __name__ == "__main__":
    import cv2

    image = cv2.imread('../data/user1/user1_0.jpg')
    face_hash = get_face_hash(image)
    print(face_hash)
