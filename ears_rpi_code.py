#!/usr/bin/env python3

#Import packages
import os
import argparse
import sys

from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep

import tensorflow as tf
import cv2
import numpy as np

import serial
import pandas as pd
import tkinter

#Import Motor Class

dfLabels = pd.read_csv('labelList.csv')
root = tkinter.Tk()

ser = serial.Serial('/dev/ttyUSB0', 115200)
sleep(0.5)

# Set up camera constants
IM_WIDTH = root.winfo_screenwidth()
IM_HEIGHT = root.winfo_screenheight()

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'

#### Initialize TensorFlow model ####

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#### Initialize other parameters ####

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Define outside box coordinates (top left and bottom right)
TL_left = (int(IM_WIDTH*0.001),int(IM_HEIGHT*0.2)) #0.001 / 0.2
BR_left = (int(IM_WIDTH*0.2),int(IM_HEIGHT*0.94)) #0.2 / 0.94

TL_right = (int(IM_WIDTH*0.999),int(IM_HEIGHT*0.2)) #0.999/0.6
BR_right = (int(IM_WIDTH*0.8),int(IM_HEIGHT*0.94)) #0.8 / 0.94

TL_middle= (int(IM_WIDTH*0.203),int(IM_HEIGHT*0.2)) #0.351/0.2
BR_middle = (int(IM_WIDTH*0.797),int(IM_HEIGHT*0.94)) #0.649 / 0.94


# Initialize control variables used for the detector
detected_left = False
detected_right = False
detected_middle = False

left_counter = 0
right_counter = 0
middle_counter = 0

pause = 0
pause_counter = 0

# Initialize control variables used for the middle check
middle_counter_MC = 0
error_counter_MC = 0


yourLabel = None
yourLabel_MC = None

#### Detection function ####
# Check For Special Classes 
def checkMiddle(frame):
    global yourLabel_MC
    global middle_counter_MC

    print("In check middle.")

    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    if (((int(classes[0][0]) == 1) or (int(classes[0][0]) == 2) or (int(classes[0][0]) == 3) or (int(classes[0][0]) == 4) or (int(classes[0][0]) == 6) or (int(classes[0][0]) == 7) or (int(classes[0][0]) == 8) or (int(classes[0][0]) == 9)) or (int(classes[0][0]) == 10)or (int(classes[0][0]) == 11) or (int(classes[0][0]) == 13) or (int(classes[0][0]) == 14) or (int(classes[0][0]) == 15)or (int(classes[0][0]) == 17) or (int(classes[0][0]) == 18) or (int(classes[0][0]) == 88)) and (pause == 0):
            x = int(((boxes[0][0][1]+boxes[0][0][3])/2)*IM_WIDTH)
            y = int(((boxes[0][0][0]+boxes[0][0][2])/2)*IM_HEIGHT)

            yourLabel = dfLabels.loc[dfLabels['number'] == (classes[0][0]), ['label']].values[0]
            yourLabel = str(yourLabel)
            # Draw a circle at center of object
            cv2.circle(frame,(x,y), 5, (75,13,180), -1)
            cv2.putText(frame, yourLabel, (x,y), font, 0.5, (75,13,180), 1, cv2.LINE_AA)

            # If object is in middle box, increment middle counter variable
            if ((x > TL_middle[0]) and (x < BR_middle[0]) and (y > TL_middle[1]) and (y < BR_middle[1])):
                middle_counter_MC = middle_counter_MC + 1
                print("Middle Counter:", middle_counter_MC)
            else:
                error_counter_MC = error_counter_MC + 1
                print("Error Counter:", error_counter_MC)

    if middle_counter_MC > 1:
        print("There is a", yourLabel_MC, "in your way.")
        return

    if error_counter_MC > 1:
        print("No object in the middle.")
        return

def main_detector(frame):
    # Use globals for the control variables so they retain their value after function exits
    global detected_left, detected_right, detected_middle
    global left_counter, right_counter, middle_counter
    global pause, pause_counter
    global yourLabel

    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
   
    cv2.rectangle(frame,TL_right,BR_right,(255,20,20),3)
    cv2.rectangle(frame,TL_left,BR_left,(20,20,255),3)
    cv2.rectangle(frame,TL_middle,BR_middle,(127,0,255),3)


    if ser.in_waiting > 0:
        earValue = ser.readline().decode('utf-8').rstrip()
        if earValue == "Turn Left!":
            checkMiddle(frame)
        elif earValue == "Turn Right!":
            checkMiddle(frame)
    else:
        if (((int(classes[0][0]) == 1) or (int(classes[0][0]) == 2) or (int(classes[0][0]) == 3) or (int(classes[0][0]) == 4) or (int(classes[0][0]) == 6) or (int(classes[0][0]) == 7) or (int(classes[0][0]) == 8) or (int(classes[0][0]) == 9)) or (int(classes[0][0]) == 10)or (int(classes[0][0]) == 11) or (int(classes[0][0]) == 13) or (int(classes[0][0]) == 14) or (int(classes[0][0]) == 15)or (int(classes[0][0]) == 17) or (int(classes[0][0]) == 18) or (int(classes[0][0]) == 88)) and (pause == 0):
            x = int(((boxes[0][0][1]+boxes[0][0][3])/2)*IM_WIDTH)
            y = int(((boxes[0][0][0]+boxes[0][0][2])/2)*IM_HEIGHT)

            yourLabel = dfLabels.loc[dfLabels['number'] == (classes[0][0]), ['label']].values[0]
            yourLabel = str(yourLabel)
            # Draw a circle at center of object
            cv2.circle(frame,(x,y), 5, (75,13,180), -1)
            cv2.putText(frame, yourLabel, (x,y), font, 0.5, (75,13,180), 1, cv2.LINE_AA)

            # If object is in left box, increment left counter variable
            if ((x > TL_left[0]) and (x < BR_left[0]) and (y > TL_left[1]) and (y < BR_left[1])):
                left_counter = left_counter + 1
                print("Left Counter:", left_counter)

            # If object is in right box, increment right counter variable
            if ((x < TL_right[0]) and (x > BR_right[0]) and (y > TL_right[1]) and (y < BR_right[1])):
                right_counter = right_counter + 1
                print("Right Counter:", right_counter)

            # If object is in middle box, increment middle counter variable
            if ((x > TL_middle[0]) and (x < BR_middle[0]) and (y > TL_middle[1]) and (y < BR_middle[1])):
                middle_counter = middle_counter + 1
                print("Middle Counter:", middle_counter)
        
    if left_counter > 3:
        detected_left = True
        print("There is a", yourLabel, "coming from your left.")
        #ser.write(str.encode('E'))
        
        left_counter = 0
        right_counter = 0
        middle_counter = 0
        pause = 1

        sleep(2)
        checkMiddle(frame)

    if right_counter > 3:
        detected_right = True
        print("There is a", yourLabel, "coming from your right.")
        #ser.write(str.encode('T'))
        
        left_counter = 0
        right_counter = 0
        middle_counter = 0
        pause = 1

        sleep(2)
        checkMiddle(frame)

    if middle_counter > 5:
        detected_middle = True
        #ser.write(str.encode('O')) 
        #Kod
        
        left_counter = 0
        right_counter = 0
        middle_counter = 0
        pause = 1

    if pause == 1:
        pause_counter = pause_counter +1
        if pause_counter > 30:
            pause = 0
            pause_counter = 0
            detected_left = False
            detected_right = False
            detected_middle = False       

    return frame

### Picamera ###
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    # Continuously capture frames and perform object detection on them
    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)

        # Pass frame into pet detection function
        frame = main_detector(frame)

        # Draw FPS
        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # FPS calculation
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()
        
cv2.destroyAllWindows()
