# 1. Thêm các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist

# 2. Load dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_val, y_val = X_train[50000:60000, :], y_train[50000:60000]
X_train, y_train = X_train[:50000, :], y_train[:50000]
print("Kích thước dữ liệu huấn luyện:", X_train.shape)

# 3. Reshape lại dữ liệu cho đúng kích thước mà Keras yêu cầu
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255.0
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255.0

# 4. One-hot encoding cho label (Y)
Y_train = to_categorical(y_train, 10)
Y_val = to_categorical(y_val, 10)
Y_test = to_categorical(y_test, 10)
print("Dữ liệu y ban đầu:", y_train[0])
print("Dữ liệu y sau one-hot encoding:", Y_train[0])

# 5. Định nghĩa model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

# 6. Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. Thực hiện train model
H = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=32, epochs=10, verbose=1)

# 8. Vẽ đồ thị loss và accuracy
fig = plt.figure()
numOfEpoch = 10
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='Training Loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='Validation Loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='Training Accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()

# 9. Đánh giá model trên test set
score = model.evaluate(X_test, Y_test, verbose=0)
print("Đánh giá trên test set:", score)

# 10. Dự đoán ảnh
plt.imshow(X_test[61].reshape(28, 28), cmap='gray')
plt.title("Ảnh gốc")
plt.show()

y_predict = model.predict(X_test[61].reshape(1, 28, 28, 1))
print("Xác suất dự đoán:", y_predict)
print("Giá trị dự đoán (nhãn):", np.argmax(y_predict))
