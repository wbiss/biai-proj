import numpy as np
import cv2
import tensorflow as tf
import argparse

facialExpressions = ["Neutral -.-", "Happy  :)", "Sad :< ", "Surprise :O ", "Angry >:O "]


def Program(model, cameraNumber, scaleFactor, minNeighbors, minScaleX, minScaleY):
    faceDetector = cv2.CascadeClassifier('haar_cascade_face_detection.xml')
    camera = cv2.VideoCapture(cameraNumber)
    facialExpressionModel = tf.keras.models.load_model(model)

    while True:
        _, frame = camera.read()
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detectedFaces = faceDetector.detectMultiScale(grayFrame, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                                      minSize=(minScaleX, minScaleY))
        cv2.rectangle(frame, (0, 0), (140, 40), (27, 84, 215), -1)
        for xCoordinate, yCoortinate, width, height in detectedFaces:
            cv2.rectangle(frame, (xCoordinate, yCoortinate), (xCoordinate + width, yCoortinate + height), (27, 84, 215),
                          2)
            greyScaleFace = getGrayFace(grayFrame, xCoordinate, yCoortinate, width, height)

            faceExpression = predictFaceExpression(greyScaleFace, facialExpressionModel)

            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame, faceExpression, (20, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_8)

        cv2.imshow('ExpressionDetection', frame)
        if cv2.waitKey(25) != -1:
            break

    camera.release()
    cv2.destroyAllWindows()


def getGrayFace(grayImage, xCoordinate, yCoortinate, width, height):
    croppedFace = grayImage[yCoortinate + 5: yCoortinate + height - 5, xCoordinate + 15:xCoordinate + width - 15]
    resizedFace = cv2.resize(croppedFace, (48, 48))
    greyScaleFace = resizedFace / 255.0
    return greyScaleFace


def predictFaceExpression(greyScaleFace, model):
    preparedFaceForModel = np.array([greyScaleFace.reshape((48, 48, 1))])
    predictionRaw = model.predict(preparedFaceForModel)
    maxValue = predictionRaw.argmax()
    detectedExpression = facialExpressions[maxValue]
    return detectedExpression


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="path to trained model", nargs=1, type=str, default="expression.model")
    parser.add_argument("-o", help="camera number", nargs=1, type=int, default=0)
    parser.add_argument("-s", help="scale factor", nargs=1, type=float, default=1.3)
    parser.add_argument("-n", help="minNeighbors", nargs=1, type=int, default=5)
    parser.add_argument("-x", help="minScaleX", nargs=1, type=int, default=50)
    parser.add_argument("-y", help="minScaleY", nargs=1, type=int, default=50)

    args = parser.parse_args()
    if (isinstance(args.m, list)):
        args.m = args.m[0]
    if (isinstance(args.o, list)):
        args.o = args.o[0]
    if (isinstance(args.s, list)):
        args.s = args.s[0]
    if (isinstance(args.n, list)):
        args.n = args.n[0]
    if (isinstance(args.x, list)):
        args.x = args.x[0]
    if (isinstance(args.y, list)):
        args.y = args.y[0]
    Program(args.m, args.o, args.s, args.n, args.x, args.y)