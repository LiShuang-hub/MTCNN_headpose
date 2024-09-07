#!/usr/bin/env python3
import os
import cv2
import sys
import dlib
import argparse
import numpy as np
import serial  # 导入pyserial模块

# import Face Recognition
import face_recognition

# helper modules
from drawFace import draw
import reference_world as world

PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")

if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--focal", type=float, help="Calibrated Focal Length of the camera")
parser.add_argument("-s", "--camsource", type=int, default=0, help="Enter the camera source")
parser.add_argument("-p", "--port", type=str, help="Bluetooth serial port (e.g., COM3 or /dev/rfcomm0)", required=True)
args = vars(parser.parse_args())

face3Dmodel = world.ref3DModel()

def send_command(bluetooth, command):
    bluetooth.write((command + '\n').encode())

def get_control_command(yaw, pitch, roll):
    if pitch > -2.50 or pitch < -3.0:
        return 'STOP'
    elif -3.0 < pitch < -2.5:
        return 'FORWARD!'
    # elif yaw > 15:
    #     return 'RIGHT'
    elif roll > 1.6:
        return 'LEFT!'
    elif roll < 1.4:
        return 'RIGHT!'
    return 'STOP'

def main():
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # 初始化蓝牙串口连接
    bluetooth = serial.Serial(args["port"], baudrate=9600, timeout=1)

    cap = cv2.VideoCapture(args["camsource"])

    while True:
        GAZE = "Face Not Found"
        ret, img = cap.read()
        if not ret:
            print(f'[ERROR - System] Cannot read from source: {args["camsource"]}')
            break

        faces = face_recognition.face_locations(img, model="cnn")

        for face in faces:
            x = int(face[3])
            y = int(face[0])
            w = int(abs(face[1] - x))
            h = int(abs(face[2] - y))
            u = int(face[1])
            v = int(face[2])

            newrect = dlib.rectangle(x, y, u, v)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), newrect)

            draw(img, shape)

            refImgPts = world.ref2dImagePoints(shape)

            height, width, channels = img.shape
            focalLength = args["focal"] * width
            cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

            mdists = np.zeros((4, 1), dtype=np.float64)

            success, rotationVector, translationVector = cv2.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)

            axis = np.float32([[200, 0, 0], [0, 200, 0], [0, 0, 200]])
            imgPts, jac = cv2.projectPoints(axis, rotationVector, translationVector, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            imgPts = imgPts.astype(int)  # Convert to integer
            cv2.line(img, p1, tuple(imgPts[0].ravel()), (0, 0, 255), 3) # X-axis (red)
            cv2.line(img, p1, tuple(imgPts[1].ravel()), (0, 255, 0), 3) # Y-axis (green)
            cv2.line(img, p1, tuple(imgPts[2].ravel()), (255, 0, 0), 3) # Z-axis (blue)

            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x_angle = np.arctan2(Qx[2][1], Qx[2][2])
            y_angle = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1]) + (Qy[2][2] * Qy[2][2])))
            z_angle = np.arctan2(Qz[0][0], Qz[1][0])
            print(f"Pitch: {x_angle}, Yaw: {y_angle}, Roll: {z_angle}")

            command = get_control_command(y_angle, x_angle, z_angle)
            send_command(bluetooth, command)

            if angles[1] < -15:
                GAZE = "Looking: Left"
            elif angles[1] > 15:
                GAZE = "Looking: Right"
            else:
                GAZE = "Forward"

            if angles[0] < -15:
                GAZE += ", Head: Down"
            elif angles[0] > 15:
                GAZE += ", Head: Up"
            elif -15 <= angles[0] <= 15:
                GAZE += ", LF"

            pitch_text = f"Pitch: {np.degrees(x_angle):.2f}"
            yaw_text = f"Yaw: {np.degrees(y_angle):.2f}"
            roll_text = f"Roll: {np.degrees(z_angle):.2f}"
            cv2.putText(img, pitch_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(img, yaw_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(img, roll_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(img, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        cv2.imshow("Head Pose", img)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
