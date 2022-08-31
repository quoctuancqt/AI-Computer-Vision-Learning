# Import required modules
from cProfile import label
import enum
from ntpath import join
from os import listdir
from os.path import isfile, join
import cv2 as cv
import time
import argparse
import numpy as np
import face_recognition
import uuid

parser = argparse.ArgumentParser(
    description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument(
    '--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--mode", default="cascadeclassifier")

args = parser.parse_args()
args = parser.parse_args()
args = parser.parse_args()

output_path = "./data/"
padding = 20
# age model
# model structure: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt
# pre-trained weights: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel
age_model = cv.dnn.readNetFromCaffe(
    "./models/age.prototxt", "vdex_chalearn_iccv2015.caffemodel")
# gender model
# model structure: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt
# pre-trained weights: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel
gender_model = cv.dnn.readNetFromCaffe("./models/gender.prototxt", "./models/gender.caffemodel")
# face model
haar_detector = cv.CascadeClassifier("./models/haarcascade_frontalface_default.xml")


def detect_faces(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = haar_detector.detectMultiScale(gray, 1.3, 5)
    return faces


model = cv.face.LBPHFaceRecognizer_create()

only_files = []
for f in listdir(output_path):
    if isfile(join(output_path, f)):
        only_files.append([0, 0, False, f])


def model_train():
    global model
    train_data, labels = [], []
    for i, files in enumerate(only_files):
        image_path = output_path + only_files[i][3]
        images = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        train_data.append(np.asarray(images, dtype=np.uint8))
        labels.append(i)
    labels = np.asarray(labels, dtype=np.int32)
    if len(train_data) > 0:
        model.train(np.asarray(train_data), np.asarray(labels))
    print("Model trained succesfully")


def recognize(detected_face):
    global only_files, model, output_path
    detected_face = cv.cvtColor(detected_face, cv.COLOR_BGR2GRAY)
    ####store data#######
    cv.imwrite("{}{}.jpg".format(output_path, uuid.uuid4()), detected_face)
    #####################

    ####predict##########
    try:
        idx, conf = model.predict(detected_face)
        print(conf)
        if conf < 500:
            confidence = int(100*(1-(conf)/300))
            # display_string = str(confidence) + " % Confidence it is user"

            # cv.putText(frame, display_string, (100, 120),
            #            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if confidence > 80:
            if only_files[idx][2] == False:
                only_files[idx][2] = True
                only_files[idx][0] = time.time()
            else:
                only_files[idx][1] = time.time()

            tt = only_files[idx][1] - only_files[idx][0]
            if tt > 0:
                cv.putText(frame, "{} seconds".format(round(tt)), (100, 120),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(frame, "KNOWN", (250, 450),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(frame, "UNKNOWN", (250, 450),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except:
        pass
    return frame
    #####################


def recognize_using_cascadeclassifier(frame, t):
    faces = detect_faces(frame)
    for x, y, w, h in faces:
        detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
        detected_face = cv.resize(detected_face, (224, 224))
        # recognize(detect_faces)
        img_blob = cv.dnn.blobFromImage(detected_face)
        age_model.setInput(img_blob)
        age_dist = age_model.forward()[0]
        gender_model.setInput(img_blob)
        gender_class = gender_model.forward()[0]
        # output_indexes = np.array([i for i in range(0, 101)])
        # apparent_predictions = round(np.sum(age_dist * output_indexes))
        apparent_predictions = age_dist.argmax()
        gender = 'Woman ' if np.argmax(gender_class) == 0 else 'Man'
        label = "{},{}".format(gender, apparent_predictions)

        cv.putText(frame, label, (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow("Age Gender Demo", frame)

        print("'recognize using cascadeclassifier' time : {:.3f}".format(
            time.time() - t))


faceNet = cv.dnn.readNet("./models/opencv_face_detector_uint8.pb",
                         "./models/opencv_face_detector.pbtxt")


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                         (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


def recognize_using_cnn(frame, t):
    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bbox in bboxes:
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                     max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        detected_face = cv.resize(face, (224, 224))
        img_blob = cv.dnn.blobFromImage(detected_face)
        age_model.setInput(img_blob)
        age_dist = age_model.forward()[0]
        gender_model.setInput(img_blob)
        gender_class = gender_model.forward()[0]
        # output_indexes = np.array([i for i in range(0, 101)])
        # apparent_predictions = round(np.sum(age_dist * output_indexes))
        apparent_predictions = age_dist.argmax()
        gender = 'Woman ' if np.argmax(gender_class) == 0 else 'Man'
        label = "{},{}".format(gender, apparent_predictions)

        cv.putText(frameFace, label, (bbox[0], bbox[1]-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow("Age Gender Demo", frameFace)

    print("'recognize using cnn' time : {:.3f}".format(time.time() - t))


# Open a video file or an image file or a camera stream
cap = cv.VideoCapture(args.input if args.input else 0)
# Open a stream from rmtp server
# cap = cv.VideoCapture("rtmp://10.0.30.43/live/test")
while cv.waitKey(1) < 0:
    # Read frame
    t = time.time()
    # time.sleep(5)
    # model_train()
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    if args.mode == "cascadeclassifier":
        recognize_using_cascadeclassifier(frame, t)
    else:
        recognize_using_cnn(frame, t)
