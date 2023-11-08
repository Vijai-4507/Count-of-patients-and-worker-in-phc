import face_recognition
import cv2
import numpy as np
import csv
import os
import time  
from datetime import datetime


video_capture = cv2.VideoCapture(0)
known_faces_names = []
known_face_encoding = []  
students = known_faces_names.copy()
face_locations = []
face_encodings = []
face_names = []
recognized_patients = []  
recognized_workers = []   
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file = f"{current_date}.csv"
with open(csv_file, 'w+', newline='') as f:
    writer = csv.writer(f)

    
    worker_count = 0
    patient_count = 0
    time_window = 5  

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Perform face recognition
        if True:  
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
                face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
                best_match_index = np.argmin(face_distance)

                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]
                    face_names.append(name)

                    if name in students:
                        students.remove(name)
                        current_time = time.time()
                        
                        if all(current_time - entry[1] > time_window for entry in recognized_workers):
                            recognized_workers.append((name, current_time))
                            worker_count += 1
                            current_time_str = now.strftime("%H-%M-%S")
                            writer.writerow([name, current_time_str, "Worker"])
                else:
                    
                    if all(face_recognition.compare_faces([entry[0]], face_encoding)[0] == False for entry in recognized_patients):
                        recognized_patients.append((face_encoding, time.time()))
                        patient_count += 1
                        current_time = now.strftime("%H-%M-%S")
                        writer.writerow(["Unknown Patient", current_time, "Patient"])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    print("Total Worker Count:", worker_count)
    print("Total Patient Count:", patient_count)

video_capture.release()
cv2.destroyAllWindows()
