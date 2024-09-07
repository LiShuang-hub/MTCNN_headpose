#!/usr/bin/env python3
import os
import cv2
import sys
import dlib
import numpy as np
from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import serial

# import Face Recognition
import face_recognition
import argparse
# helper modules
from drawFace import draw
import reference_world as world
def rotate_image(image_np):
    # 旋转图像 90 度
    rotated_image = cv2.transpose(image_np)
    rotated_image = cv2.flip(rotated_image, flipCode=0)
    return rotated_image
parser = argparse.ArgumentParser()
#parser.add_argument("-f", "--focal", type=float, help="Calibrated Focal Length of the camera", required=True)
parser.add_argument("-p", "--port", type=str, help="Bluetooth serial port (e.g., COM3 or /dev/rfcomm0)", required=True)
args = vars(parser.parse_args())
bluetooth = serial.Serial(args["port"], baudrate=9600, timeout=1)
app = Flask(__name__)
last_command = None
PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")

if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()

face3Dmodel = world.ref3DModel()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
def resize_image(image_np, new_size=(640, 480)):
    # 调整图像大小
    resized_image = cv2.resize(image_np, new_size, interpolation=cv2.INTER_LINEAR)
    return resized_image

def send_command(bluetooth, command):
    global last_command
        # 检查当前命令是否与上一次发送的命令相同
    if command != last_command:
        bluetooth.write((command + '\n').encode())
        print(command)
        last_command = command  # 更新上一次发送的命令
    else:
        print(f"Command '{command}' is the same as the last one, not sending.")
        #return



def get_control_command(yaw, pitch, roll, gaze_direction):
    if "Looking: Right" in gaze_direction:
        return '204'  # 前进右转
    elif "Looking: Left" in gaze_direction:
        return '203'  # 前进左转
    elif "Head: Up" in gaze_direction:
        return '201'  # 直行
    elif "Head: Down" in gaze_direction:
        return '010'  # 停止
    else:
        return '010'  # 默认停止命令，确保始终返回一个有效命令




@app.route('/upload', methods=['POST'])
def upload_image():
 try:
    data = request.get_json()
    #print("request data: ", data)

    image_data = data['image']

    image_types = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_types))
    #image.show()
    #image_np = np.array(image)  # 将 PIL 图像转换为 numpy 数组
    image = rotate_image(np.array(image))

    #img = resize_image(image, new_size=(640, 480))
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #img = resize_image(img, new_size=(640, 480))
    #cv2.imshow("image1", img)

    cv2.waitKey(1)
    # print(img.shape)
    GAZE = "Face Not Found"
    faces = face_recognition.face_locations(img, model="cnn")
    #print(faces[0])
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
        focalLength = 3354.735 * width  # Use a default focal length
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
        imgPts = imgPts.astype(int)
        cv2.line(img, p1, tuple(imgPts[0].ravel()), (0, 0, 255), 3)
        cv2.line(img, p1, tuple(imgPts[1].ravel()), (0, 255, 0), 3)
        cv2.line(img, p1, tuple(imgPts[2].ravel()), (255, 0, 0), 3)
        print(img.shape)
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        x_angle = np.arctan2(Qx[2][1], Qx[2][2])
        y_angle = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1]) + (Qy[2][2] * Qy[2][2])))
        z_angle = np.arctan2(Qz[0][0], Qz[1][0])
        #print(f"Pitch: {x_angle}, Yaw: {y_angle}, Roll: {z_angle}")
        #angles_degrees = np.degrees(angles)
        if np.degrees(y_angle) > 35:
            GAZE = "Looking: Left"
        elif np.degrees(y_angle) < -20:
            GAZE = "Looking: Right"
        else:
            GAZE = "Forward"

        if np.degrees(x_angle) < -145 and np.degrees(x_angle) >-165:
            GAZE += ", Head: Up"
        else :
            GAZE += ", Head: Down"
        print(GAZE)

        command = get_control_command(y_angle, x_angle, z_angle, GAZE)
        send_command(bluetooth, command)
        pitch_text = f"Pitch: {np.degrees(x_angle):.2f}"
        yaw_text = f"Yaw: {np.degrees(y_angle):.2f}"
        roll_text = f"Roll: {np.degrees(z_angle):.2f}"
        print("P: ,Y: ,R: ",np.degrees(x_angle), np.degrees(y_angle), np.degrees(z_angle))
        cv2.putText(img, pitch_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, yaw_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, roll_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(img, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
    #cv2.imshow("image", img)

    _, buffer = cv2.imencode('.jpg', img)
    response_image = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"image": response_image, "gaze": GAZE})
 except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    context = ('cert.pem', 'key.pem')  # Paths to your certificate and key
    app.run(host="0.0.0.0", port=5000, ssl_context=context)
