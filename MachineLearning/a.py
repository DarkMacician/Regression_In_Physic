import cv2
import face_recognition

# Đọc ảnh
image_path = 'D:\MachineLearning\captured_frame_0.jpg'
image = face_recognition.load_image_file(image_path)

# Phát hiện khuôn mặt
face_locations = face_recognition.face_locations(image)

# Hiển thị số lượng khuôn mặt phát hiện được
print(f"Found {len(face_locations)} face(s)")

# Vẽ hình chữ nhật quanh các khuôn mặt
for (top, right, bottom, left) in face_locations:
    face_image = image[top-100:bottom+50, left-25:right+25]
    face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("face.jpg", face_image_bgr)

# Hiển thị ảnh với các khuôn mặt được đánh dấu
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
