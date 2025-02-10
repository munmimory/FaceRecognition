import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import numpy as np


# Load bộ nhận diện khuôn mặt Haar Cascade
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Mở webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Chiều rộng của video
cap.set(4, 480)  # Chiều cao của video
font = cv2.FONT_HERSHEY_COMPLEX

# Load mô hình đã huấn luyện
model = load_model('keras_model.h5')

# Hàm trả về tên của lớp dựa trên index
def get_className(classNo):
    if classNo == 0:
        return "Nguyet"
    elif classNo == 1:
        return "Luan"
    elif classNo == 2:
        return "Man"

# Vòng lặp chính để nhận diện khuôn mặt
while True:
    success, imgOriginal = cap.read()
    faces = facedetect.detectMultiScale(imgOriginal, 1.3, 5)

    for x, y, w, h in faces:
        # Cắt khuôn mặt
        crop_img = imgOriginal[y:y+h, x:x+w]

        # Resize và chuẩn hóa hình ảnh
        img = cv2.resize(crop_img, (224, 224))
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1  # Chuẩn hóa về [-1, 1]

        # Dự đoán lớp và độ tin cậy
        prediction = model.predict(img)
        classIndex = np.argmax(prediction)
        probabilityValue = np.max(prediction)

        # Vẽ khung và hiển thị kết quả
        cv2.rectangle(imgOriginal, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(imgOriginal, (x, y-40), (x+w, y), (0, 255, 0), -1)
        
        # Chỉ hiển thị tên mà không có tỷ lệ phần trăm
        cv2.putText(imgOriginal, f"{get_className(classIndex)}", 
                    (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    # Hiển thị kết quả trên cửa sổ
    cv2.imshow("Result", imgOriginal)

    # Thoát khỏi vòng lặp khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
