import json
import cv2
from generate_hash import get_face_hash

def register_user(username, image):
    face_hash = get_face_hash(image)
    if face_hash:
        try:
            with open('../user_data.json', 'r+') as f:
                user_data = json.load(f)
        except FileNotFoundError:
            user_data = {}
        user_data[username] = face_hash
        with open('../user_data.json', 'w') as f:
            json.dump(user_data, f)
        return True
    return False

if __name__ == "__main__":
    image = cv2.imread('../data/user1/user1_0.jpg')
    success = register_user('user1', image)
    if success:
        print("User registered successfully")
    else:
        print("User registration failed")
