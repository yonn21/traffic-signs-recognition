# Import các thư viện GUI cần thiết
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy  # Thư viện cho việc làm việc với mảng
from keras.models import load_model  # Load mô hình đã huấn luyện từ Keras

# Load mô hình đã huấn luyện để phân loại biển báo giao thông
model = load_model('train-model/traffic_classifier.h5')

# Một từ điển để đặt tên cho các lớp biển báo
classes = {1: 'Hạn chế tốc độ (20km/h)',
           2: 'Hạn chế tốc độ (30km/h)',
           3: 'Hạn chế tốc độ (50km/h)',
           4: 'Hạn chế tốc độ (60km/h)',
           5: 'Hạn chế tốc độ (70km/h)',
           6: 'Hạn chế tốc độ (80km/h)',
           7: 'Hết hạn chế tốc độ (80km/h)',
           8: 'Hạn chế tốc độ (100km/h)',
           9: 'Hạn chế tốc độ (120km/h)',
           10: 'Cấm vượt',
           11: 'Cấm vượt xe trọng tải trên 3.5 tấn',
           12: 'Ưu tiên ở ngã tư',
           13: 'Đường ưu tiên',
           14: 'Nhường đường',
           15: 'Dừng lại',
           16: 'Cấm đi xe',
           17: 'Cấm xe trọng tải > 3.5 tấn',
           18: 'Cấm đi (không vào)',
           19: 'Cảnh báo chung',
           20: 'Khúc cua nguy hiểm bên trái',
           21: 'Khúc cua nguy hiểm bên phải',
           22: 'Khúc cua kép',
           23: 'Đoạn đường gồ ghề',
           24: 'Đường trơn trượt',
           25: 'Đường hẹp bên phải',
           26: 'Công trường',
           27: 'Đèn tín hiệu giao thông',
           28: 'Đường dành cho người đi bộ',
           29: 'Đường dành cho trẻ em',
           30: 'Đường dành cho xe đạp',
           31: 'Cẩn thận tuyết/đá',
           32: 'Cẩn thận động vật hoang dã',
           33: 'Kết thúc hạn chế tốc độ và cấm vượt',
           34: 'Rẽ phải phía trước',
           35: 'Rẽ trái phía trước',
           36: 'Chỉ được đi thẳng',
           37: 'Đi thẳng hoặc rẽ phải',
           38: 'Đi thẳng hoặc rẽ trái',
           39: 'Luôn đi bên phải',
           40: 'Luôn đi bên trái',
           41: 'Bắt buộc đi vào vòng xuyến',
           42: 'Kết thúc cấm vượt',
           43: 'Kết thúc cấm vượt xe trọng tải > 3.5 tấn'}

# Khởi tạo giao diện GUI
top = tk.Tk()
top.geometry('700x600')  # Kích thước cửa sổ
top.title('PHÁT HIỆN BIỂN BÁO GIAO THÔNG')
top.configure(background='#c9fffb')  # Màu nền

label = Label(top, background='#c9fffb', font=('roboto', 15, 'bold'))
sign_image = Label(top)

# Hàm để phân loại và hiển thị kết quả phân loại biển báo từ ảnh
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = numpy.expand_dims(image, axis=0)  # Mở rộng số chiều của mảng
    image = numpy.array(image)  # Chuyển ảnh thành mảng NumPy
    print(image.shape)
    pred = numpy.argmax(model.predict(image), axis=-1)  # Dự đoán lớp
    sign = classes[pred[0] + 1]  # Lấy tên biển báo từ từ điển
    print(sign)
    # Hiển thị kết quả phân loại
    label.configure(foreground='#011638', text=sign)

# Hàm để hiển thị nút "PHÂN TÍCH" và liên kết với việc phân loại ảnh
def show_classify_button(file_path):
    classify_b = Button(top, text="PHÂN TÍCH",
                        command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156',
                         foreground='white', font=('roboto', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)  # Đặt vị trí của nút

# Hàm để tải ảnh và hiển thị nút phân loại khi ảnh được tải lên
def upload_image():
    try:
        file_path = filedialog.askopenfilename()  # Chọn tệp ảnh
        uploaded = Image.open(file_path)
        uploaded.thumbnail(
            ((top.winfo_width()/2.25), (top.winfo_height()/2.25)))  # Giới hạn kích thước ảnh
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

# Tạo nút "CHỌN ẢNH" để tải ảnh lên và gắn liên kết với hàm tải ảnh
upload = Button(top, text="CHỌN ẢNH",
                command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white',
                 font=('roboto', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)  # Đặt vị trí của nút "CHỌN ẢNH"
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="PHÁT HIỆN BIỂN BÁO GIAO THÔNG",
                pady=20, font=('roboto', 20, 'bold'))  # Tiêu đề
heading.configure(background='#c9fffb', foreground='#364156')
heading.pack()  # Đặt vị trí của tiêu đề
top.mainloop()  # Khởi chạy giao diện GUI
