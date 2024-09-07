import cv2
import os

# 相机ID，一般为0，如果有多个摄像头可以尝试不同的ID
camera_id = 0

# 保存图像的文件夹路径
save_folder = './data/calibrate_images/'

# 创建保存图像的文件夹
os.makedirs(save_folder, exist_ok=True)

# 棋盘格的行数和列数
num_rows = 6
num_cols = 7

# 设置摄像头参数
cap = cv2.VideoCapture(camera_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置摄像头的宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 设置摄像头的高度

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'p' to capture a photo of the chessboard. Press 'q' to quit.")

photo_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame")
        break

    cv2.imshow('Chessboard Capture', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):  # Press 'p' to capture a photo
        photo_count += 1
        filename = os.path.join(save_folder, f'chessboard_{photo_count}.jpg')
        cv2.imwrite(filename, frame)
        print(f"Photo {photo_count} captured.")
    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
