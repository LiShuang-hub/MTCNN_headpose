#!/usr/bin/env python3
import os
import cv2
import sys
import dlib
import argparse
import numpy as np
import serial  # 导入pyserial模块
import threading
import time
from flask import Flask, request, jsonify
import base64
import io
from PIL import Image

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
parser.add_argument("-f", "--focal", type=float, help="Calibrated Focal Length of the camera", required=True)
parser.add_argument("-p", "--port", type=str, help="Bluetooth serial port (e.g., COM3 or /dev/rfcomm0)", required=True)
args = vars(parser.parse_args())

face3Dmodel = world.ref3DModel()

last_command = None
last_send_time = 0

# 初始化Flask应用
app = Flask(__name__)

# 初始化蓝牙串口连接
bluetooth = serial.Serial(args["port"], baudrate=9600, timeout=1)


def send_command(bluetooth, command):
    global last_command, last_send_time

    current_time = time.time()

    # 检查是否已经过了一秒
    if current_time - last_send_time >= 0.2:
        if command != last_command:
            bluetooth.write((command + '\n').encode())
            print(command)
            last_command = command  # 更新上一次发送的命令
        else:
            print(f"Command '{command}' is the same as the last one, not sending.")
        last_send_time = current_time  # 更新上一次发送的时间


def get_control_command(yaw, pitch, roll, gaze_direction):
    if "Looking: Right, Head: Up" in gaze_direction:
        return '804'  # 前进右转
    elif "Looking: Left, Head: Up" in gaze_direction:
        return '803'  # 前进左转
    elif "Forward, Head: Up" in gaze_direction:
        return '801'  # 直行
    elif "Forward, Head: Down" in gaze_direction:
        return '010'  # 停止
    else:
        return '010'  # 默认停止命令，确保始终返回一个有效命令


def process_image(image, predictor):
    faces = face_recognition.face_locations(image, model="cnn")
    GAZE = "Face Not Found"
    for face in faces:
        x = int(face[3])
        y = int(face[0])
        w = int(abs(face[1] - x))
        h = int(abs(face[2] - y))
        u = int(face[1])
        v = int(face[2])

        newrect = dlib.rectangle(x, y, u, v)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        shape = predictor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), newrect)

        draw(image, shape)

        refImgPts = world.ref2dImagePoints(shape)

        height, width, channels = image.shape
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
        cv2.line(image, p1, tuple(imgPts[0].ravel()), (0, 0, 255), 3)  # X-axis (red)
        cv2.line(image, p1, tuple(imgPts[1].ravel()), (0, 255, 0), 3)  # Y-axis (green)
        cv2.line(image, p1, tuple(imgPts[2].ravel()), (255, 0, 0), 3)  # Z-axis (blue)

        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        x_angle = np.arctan2(Qx[2][1], Qx[2][2])
        y_angle = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1]) + (Qy[2][2] * Qy[2][2])))
        z_angle = np.arctan2(Qz[0][0], Qz[1][0])
        print(f"Pitch: {x_angle}, Yaw: {y_angle}, Roll: {z_angle}")

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

        command = get_control_command(y_angle, x_angle, z_angle, GAZE)
        send_command(bluetooth, command)

    return image


@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json
    image_data = data['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    processed_image = process_image(image, predictor)

    ret, jpeg = cv2.imencode('.jpg', processed_image)
    return jsonify({'result': base64.b64encode(jpeg.tobytes()).decode('utf-8')})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
