# import the necessary packages
from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True)
args = vars(ap.parse_args())

# Load model da train
model = load_model('models/sport.h5')
lb = pickle.loads(open('models/lb.pickle', "rb").read())

# Khai bao queue nhan dien
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=128)

# Khoi tao cac bien
i=0
label = "Predicting..."

# Doc video
vs = cv2.VideoCapture(args["video"])

while True:
        # Lay anh tu video
        ret, frame = vs.read()
        if not ret:
            break

        i+=1
        display = frame.copy()

        # Xu ly moi 10 frame
        if i%10==0:
                # Resize dua vao mang
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224)).astype("float32")
                frame -= mean

                # Du doan va dua ra ket qua
                preds = model.predict(np.expand_dims(frame, axis=0))[0]
                Q.append(preds)

                # Tinh trung binh cong
                results = np.array(Q).mean(axis=0)

                # Lay lop lon nhat va gan label
                i = np.argmax(results)
                label = lb.classes_[i]

        # Hien thi len video
        text = "I'm watching: {}".format(label)
        cv2.putText(display, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # show the output image
        cv2.imshow("Output", display)
        if cv2.waitKey(1) & 0xFF ==ord("q"):
                break

vs.release()