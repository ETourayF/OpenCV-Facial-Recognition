import cv2
import imutils
from imutils.video import VideoStream
import time
import numpy as np

import DNN_Detector as Dnn1

def train_classifier():
    #obtain images and there respective id's from the training set
    faces, f_id = Dnn1.labels_with_training_data('Training_images')
    #initialise lbph recogniser and train it using the image data that was just loaded
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(f_id))
    #save the resulting training data as a yml file
    recognizer.write('training_data.yml')

#function to properly extract student info from student file
def get_pair(line):
    key, sep, value = line.strip().partition(" ")
    return int(key), value

def dnn_lbph_recogniser():
    #initiate video stream
    vs = VideoStream(src=0).start()
    #create a recogniser object
    recognizer = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 15)
    #load in data from student information file and place in a dictionary
    with open("studentInfo") as fd:
        name = dict(get_pair(line) for line in fd)

    #load the saved lbph training data
    recognizer.read('training_data.yml')

    while True:
        #capture current frame and resize
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        #put frame through detect_faces() function to gety array of detections
        detected = Dnn1.detect_faces(frame)
        if not detected is None:
            #go through detections
            for (startX, startY, endX, endY) in detected:
                #cobnvert to gray image and cut out faces at each coordinate in the array
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_gray = gray_image[startY:endY, startX:endX]
                #put cut out into recogniser to obtain predicted label and confidence
                label, confidence = recognizer.predict(roi_gray)

                #create labels for id name and confidence
                #check label against data in student dictionary to obtain name associated with label
                detected_id = "ID: " + str(label)
                detected_name = "name: " + name[label]
                #percentage value is a float, format to display as percentage
                conf_perc = "%.2f" % round(100-float(confidence), 2)
                conf_lbl = conf_perc + "%"

                # draw rectangle
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 1)
                # display id label
                cv2.putText(frame, detected_id, (startX + 3, startY + 8), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # display name label
                cv2.putText(frame, detected_name, (startX, endY + 8), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                # display confidence label
                if float(conf_perc) > 50:
                    cv2.putText(frame, conf_lbl, (startX, endY + 17), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1,
                                cv2.LINE_AA)
                else:
                    cv2.putText(frame, conf_lbl, (startX, endY + 17), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1,
                                cv2.LINE_AA)

        #loop until user presses x
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            break

def capture_training_data():
    Dnn1.capture_training_data(28137852)
    train_classifier()

'''only run capture training data function when registering a new face'''
#capture_training_data()
dnn_lbph_recogniser()

