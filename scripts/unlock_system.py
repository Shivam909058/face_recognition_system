import json
import cv2
from generate_hash import get_face_hash


def unlock_system(image):
    face_hash = get_face_hash(image)
    if not face_hash:
        return "Access Denied!"

    try:
        with open('../user_data.json', 'r') as f:
            user_data = json.load(f)
    except FileNotFoundError:
        return "No users registered."

    for username, stored_hash in user_data.items():
        if face_hash == stored_hash:
            return f"Welcome {username}!"
    return "Access Denied!"


if __name__ == "__main__":
    image = cv2.imread('../data/user1/user1_0.jpg')
    result = unlock_system(image)
    print(result)
