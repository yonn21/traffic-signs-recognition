# Import các thư viện cần thiết
from sklearn.metrics import accuracy_score  # Thư viện để tính độ chính xác
import numpy as np  # Thư viện cho việc làm việc với mảng
import pandas as pd  # Thư viện để làm việc với dữ liệu dạng bảng
import matplotlib.pyplot as plt  # Thư viện để vẽ đồ thị
import cv2  # Thư viện OpenCV cho xử lý ảnh
import tensorflow as tf  # Thư viện TensorFlow cho học máy và mạng neural
from PIL import Image  # Thư viện Pillow cho xử lý hình ảnh
import os  # Thư viện để làm việc với hệ thống tệp
# Thư viện cho việc chia dữ liệu huấn luyện và kiểm tra
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical  # Thư viện Keras cho mã hóa one-hot
# Thư viện Keras cho mô hình tuần tự và tải mô hình
from keras.models import Sequential, load_model
# Các lớp mạng trong Keras
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


data = []  # Mảng chứa dữ liệu ảnh
labels = []  # Mảng chứa nhãn tương ứng với dữ liệu ảnh
classes = 43  # Tổng số lớp (biển báo giao thông)
cur_path = os.getcwd()  # Lấy đường dẫn thư mục làm việc hiện tại

# Thu thập dữ liệu ảnh và nhãn tương ứng
for i in range(classes):
    # Đường dẫn đến thư mục của lớp i
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)  # Liệt kê tất cả các tệp trong thư mục

    for a in images:
        try:
            image = Image.open(path + '\\' + a)  # Mở ảnh
            image = image.resize((30, 30))  # Điều chỉnh kích thước ảnh
            image = np.array(image)  # Chuyển ảnh thành mảng NumPy
            data.append(image)  # Thêm ảnh vào mảng dữ liệu
            labels.append(i)  # Thêm nhãn tương ứng vào mảng nhãn
        except:
            print("Error loading image")  # Báo lỗi nếu không thể mở ảnh

# Chuyển đổi các danh sách thành mảng NumPy
data = np.array(data)
labels = np.array(labels)

# In ra kích thước của mảng dữ liệu và mảng nhãn
print(data.shape, labels.shape)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape,
      y_test.shape)  # In ra kích thước các tập dữ liệu

# Chuyển đổi nhãn thành mã hóa one-hot
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Xây dựng mô hình mạng neural
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5),
          activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Biên dịch mô hình
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

epochs = 15  # Số lần lặp qua toàn bộ tập dữ liệu huấn luyện
history = model.fit(X_train, y_train, batch_size=32,
                    epochs=epochs, validation_data=(X_test, y_test))  # Huấn luyện mô hình

model.save("traffic_classifier.h5")  # Lưu mô hình đã huấn luyện vào tệp traffic_classifier.h5

# Vẽ biểu đồ độ chính xác và hàm mất mát qua các epochs
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Kiểm tra độ chính xác trên tập dữ liệu kiểm tra
y_test = pd.read_csv('Test.csv')  # Đọc dữ liệu kiểm tra từ tệp CSV

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))

X_test = np.array(data)

pred = model.predict_classes(X_test)  # Dự đoán lớp của ảnh trong tập kiểm tra

# Tính độ chính xác trên dữ liệu kiểm tra và in ra
print(accuracy_score(labels, pred))
