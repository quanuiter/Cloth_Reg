from google.colab import files
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tải file ảnh từ máy tính
uploaded = files.upload()  # Chọn ảnh từ máy tính để tải lên

# Lấy đường dẫn file ảnh đã tải
image_path = list(uploaded.keys())[0]
print("File đã tải lên:", image_path)

# Đọc ảnh
image = Image.open(image_path).convert('L')  # Chuyển ảnh sang grayscale
# Resize ảnh về 28x28 pixels
image = image.resize((28, 28))

# Hiển thị ảnh đã xử lý
plt.imshow(image, cmap='gray')
plt.title("Ảnh sau khi xử lý")
plt.show()

# Chuyển ảnh thành numpy array và chuẩn hóa
image_array = np.array(image).astype('float32') / 255.0  # Giá trị pixel trong khoảng 0-1
image_array = image_array.reshape(1, 28, 28, 1)  # Thêm batch dimension (1 mẫu)

# Label description
label_description = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Dự đoán xác suất
y_predict = model.predict(image_array.reshape(1, 28, 28, 1))

# Chuyển xác suất thành phần trăm và làm tròn
predictions_percent = (y_predict[0] * 100).round(3)

# Tạo bảng kết quả
prediction_table = pd.DataFrame({
    "Label": label_description,
    "Xác suất (%)": predictions_percent
})

# In bảng
print(prediction_table)
print("Giá trị dự đoán (nhãn):", np.argmax(y_predict), label_description[np.argmax(y_predict)])
