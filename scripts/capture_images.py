import cv2
import os


def capture_images(username, save_path='data/'):
    cap = cv2.VideoCapture(0)
    count = 0
    user_dir = os.path.join(save_path, username)
    os.makedirs(user_dir, exist_ok=True)

    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Capturing Images', frame)
        cv2.imwrite(os.path.join(user_dir, f'{username}_{count}.jpg'), frame)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


capture_images('user1')
