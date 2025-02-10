import cv2
import numpy as np
from keras.models import load_model
import csv
import os

# Đường dẫn tới file cascade nhận diện khuôn mặt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load mô hình đã huấn luyện
model = load_model('keras_model.h5')

# Đọc dữ liệu khách hàng từ file CSV
def read_customer_data():
    customers = {}
    if os.path.exists('customer_data.csv'):
        with open('customer_data.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 4:  # Đảm bảo có đủ 4 cột (phone_number, name, visit_count, tier)
                    customers[row[0]] = {  # Dùng số điện thoại hoặc ID là key
                        'name': row[1],
                        'visit_count': int(row[2]),  # Đảm bảo rằng đây là số
                        'tier': row[3]
                    }
    return customers

# Lưu dữ liệu khách hàng vào CSV
def save_customer_data(customers):
    with open('customer_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for phone_number, data in customers.items():
            tier = assign_tier(data['visit_count'])  # Phân loại hạng khách hàng dựa trên số lần đến
            writer.writerow([phone_number, data['name'], data['visit_count'], tier])  # Lưu hạng vào CSV

# Phân loại khách hàng dựa trên số lần đến
def assign_tier(visit_count):
    if visit_count < 20:
        return 'Silver'
    elif visit_count <= 50:
        return 'Gold'
    else:
        return 'Platinum'

# Cập nhật số lần đến của khách hàng
def update_customer_visit(phone_number):
    customers = read_customer_data()
    
    if phone_number in customers:
        customers[phone_number]['visit_count'] += 1
    else:
        print("Customer not registered.")
    
    save_customer_data(customers)

# Nhận diện và phân tích khuôn mặt từ camera
def recognize_face():
    # Mở camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Unable to open camera!")
        return
    
    processed_faces = set()  # Để theo dõi các khuôn mặt đã được xử lý
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read frame from the camera!")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face_id = (x, y, w, h)
            
            if face_id not in processed_faces:  # Kiểm tra xem khuôn mặt đã được xử lý chưa
                processed_faces.add(face_id)
                
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))  # Chỉnh kích thước thành 224x224 để phù hợp với mô hình
                
                # Chuẩn hóa và chuẩn bị ảnh cho mô hình
                face_array = np.asarray(face, dtype=np.float32) / 255.0
                face_array = np.expand_dims(face_array, axis=0)
                face_array = np.expand_dims(face_array, axis=-1)  # Đảm bảo định dạng phù hợp (224, 224, 1) cho ảnh grayscale
                
                # Dự đoán khuôn mặt
                prediction = model.predict(face_array)
                max_index = np.argmax(prediction[0])
                predicted_name = get_class_name(max_index)  # Lấy tên khách hàng từ chỉ số lớp dự đoán

                # Kiểm tra nếu khuôn mặt đã nhận diện và có thông tin khách hàng
                customers = read_customer_data()
                customer_info = customers.get(predicted_name, {'name': 'Unknown', 'tier': 'Not Available', 'visit_count': 0})

                # Hiển thị tên và hạng khách hàng
                cv2.putText(frame, f"{customer_info['name']} - {customer_info['tier']}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Cập nhật số lần đến của khách hàng
                update_customer_visit(predicted_name)

        # Hiển thị video
        cv2.imshow('Face Recognition', frame)
        
        # Dừng khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Hàm trả về tên của lớp dựa trên index
def get_class_name(classNo):
    # Đảm bảo rằng classNo phải khớp với các tên trong cơ sở dữ liệu
    if classNo == 0:
        return "12345"  # ID khách hàng Nguyet
    elif classNo == 1:
        return "11111"  # ID khách hàng Luan
    elif classNo == 2:
        return "22222"  # ID khách hàng Man
    return "Unknown"  # Nếu không tìm thấy tên

# Gọi hàm nhận diện
recognize_face()
