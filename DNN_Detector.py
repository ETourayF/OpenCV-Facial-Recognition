import cv2
import os
import imutils
from imutils.video import VideoStream
import numpy as np
import time

#load pre-trained serialised caffe and prototext models
caffe = "res10_300x300_ssd_iter_140000.caffemodel"
prototext = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(prototext, caffe)

#this function will be responsible for detecting faces in the input image
def detect_faces(frame):
    # extract dimensions and resize image to 300x300 and convert to blob from image
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    #initialise an array that will be used to store faces
    face_detections = []

    #push the blob through the network and to get detections and confidence values
    net.setInput(blob)
    detections = net.forward()

    #go through detections
    for i in range(0, detections.shape[2]):
        #get confidence value associated with each detection
        confidence = detections[0, 0, i, 2]
        #only trust detections with a confidence greater than 0.5
        #this can be any value, depends on what your threshold for accuracy is
        if confidence < 0.5:
            continue

        #compute coordinates for the detection
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #convert from numpy array to integer
        return_det = box.astype("int")
        face_detections.append(return_det)

    return face_detections

#function for capturing and saving training data
def capture_training_data(s_id):
    #initialise a video stream
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    #variabble to keep track of how many frames have been captured
    frame_count = 0

    #loop until 1000 images have been captured
    while frame_count <= 1000:
        #capture current frame
        frame = vs.read()
        #dynamic save name to prevent each from overwriting the last
        save_name = "T_image " + str(frame_count) + "_" + str(s_id) + ".png"

        #resize frame and convert it to a gray image
        frame = imutils.resize(frame, width=400)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #put frame through detect_faces() function to obtain detections
        #and where on the frame they're located (coordinates)
        detected = detect_faces(frame)

        #go through array of detections
        for (startX, startY, endX, endY) in detected:
            #cut out faces on the gray image at the coordinates where the detection lies
            roi_gray = gray_image[startY:endY, startX:endX]
            #draw rectangle around the detection and display it
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 1)
            cv2.imshow("roi gray", frame)
            #increment frame count
            frame_count += 1

            #path in which to save the frame
            path = "Training_images" + "/" + str(s_id)
            #make sure path exists, if not make a new one
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                cv2.imwrite(os.path.join(path, save_name), roi_gray)

        #terminate if user presses x
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            break

#function to load each training image with its associated id
def labels_with_training_data(directory):
    faces = []
    f_id = []

    #go through every file in the training data directory
    for path, subdirnames, filenames in os.walk(directory):
        #exclude system files
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")
                continue

            #id will be the name of the file in which the image is saved
            image_id = os.path.basename(path)
            img_path = os.path.join(path, filename)

            #print path and id to terminal and load in the image
            print("img_path:", img_path)
            print("id:", image_id)
            raw_roi_gray = cv2.imread(img_path)
            #make sure that the image was successfully loaded
            if raw_roi_gray is None:
                print("Image not loaded properly")
                continue

            #convert to gray image
            roi_gray = cv2.cvtColor(raw_roi_gray, cv2.COLOR_BGR2GRAY)
            #append the face to an array and append the id to a separate array
            faces.append(roi_gray)
            f_id.append(int(image_id))

    return faces, f_id

