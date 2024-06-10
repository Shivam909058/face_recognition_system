import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import dlib
import numpy as np
import os
import pickle

# Initialize the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Path to save registered user data
user_data_path = 'new_users.pkl'

# Load registered users data if it exists
if os.path.exists(user_data_path):
    with open(user_data_path, 'rb') as f:
        registered_users = pickle.load(f)
else:
    registered_users = {}

def get_face_encoding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) > 0:
        shape = predictor(gray, rects[0])
        face_descriptor = face_rec.compute_face_descriptor(image, shape)
        return np.array(face_descriptor)
    else:
        return None

def register_face():
    global registered_users

    username = simpledialog.askstring("Input", "Enter username:", parent=root)
    if username is None:
        return

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open video device")

        while True:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to capture image")
            cv2.imshow("Register - Press 'q' to capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        face_encoding = get_face_encoding(frame)
        if face_encoding is not None:
            registered_users[username] = face_encoding
            with open(user_data_path, 'wb') as f:
                pickle.dump(registered_users, f)
            messagebox.showinfo("Success", f"Face registered for {username}!")
        else:
            messagebox.showerror("Error", "No face detected. Please try again.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def recognize_face():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open video device")

        while True:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to capture image")
            cv2.imshow("Recognize - Press 'q' to check", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        face_encoding = get_face_encoding(frame)
        if face_encoding is not None:
            for username, reg_encoding in registered_users.items():
                distance = np.linalg.norm(reg_encoding - face_encoding)
                if distance < 0.6:
                    messagebox.showinfo("Access Granted", f"Welcome back, {username}!")
                    return
            messagebox.showerror("Access Denied", "Face not recognized. Access denied.")
        else:
            messagebox.showerror("Error", "No face detected. Please try again.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def main():
    global root
    root = tk.Tk()
    root.title("Facial Recognition Lock")

    tk.Button(root, text="Register Face", command=register_face).pack(pady=10)
    tk.Button(root, text="Unlock", command=recognize_face).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()

# thanks for watching
