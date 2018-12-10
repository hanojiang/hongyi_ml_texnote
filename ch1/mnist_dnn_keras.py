from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,activations
from keras.utils import np_utils
import numpy as np

def load_data():
    (x_train,y_train),(x_test,y_test)=mnist.load_data()#60000 10000
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train/255.
    x_test = x_test/255.
    y_train = np_utils.to_categorical(y_train,10)
    y_test = np_utils.to_categorical(y_test,10)
    return (x_train,y_train),(x_test,y_test)


(x_train,y_train),(x_test,y_test)=load_data()

model = Sequential()
model.add(Dense(512,input_shape=(28*28,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=64,epochs=20,verbose=1, validation_split=0.05)
loss,acc = model.evaluate(x_test,y_test)
print('the loss of test data is' ,loss)
print('the accuracy of test data is' ,acc)

/\*Train on 57000 samples, validate on 3000 samples
/\*Epoch 1/20
/\*57000/57000 [==============================] - 14s 242us/step - loss: 0.2020 - acc: 0.9386 - val_loss: 0.1001 - val_acc: 0.9683
/\*Epoch 2/20
/\*57000/57000 [==============================] - 13s 234us/step - loss: 0.0790 - acc: 0.9750 - val_loss: 0.0873 - val_acc: 0.9753
/\*Epoch 3/20
/\*57000/57000 [==============================] - 13s 233us/step - loss: 0.0532 - acc: 0.9823 - val_loss: 0.0649 - val_acc: 0.9810
/\*Epoch 4/20
/\*57000/57000 [==============================] - 13s 234us/step - loss: 0.0388 - acc: 0.9869 - val_loss: 0.0718 - val_acc: 0.9813
/\*Epoch 5/20
/\*57000/57000 [==============================] - 14s 239us/step - loss: 0.0320 - acc: 0.9891 - val_loss: 0.0834 - val_acc: 0.9777
/\*Epoch 6/20
/\*57000/57000 [==============================] - 13s 234us/step - loss: 0.0254 - acc: 0.9918 - val_loss: 0.0702 - val_acc: 0.9817
/\*Epoch 7/20
/\*57000/57000 [==============================] - 14s 240us/step - loss: 0.0237 - acc: 0.9920 - val_loss: 0.0900 - val_acc: 0.9810
/\*Epoch 8/20
/\*57000/57000 [==============================] - 13s 236us/step - loss: 0.0184 - acc: 0.9937 - val_loss: 0.0782 - val_acc: 0.9830
/\*Epoch 9/20
/\*57000/57000 [==============================] - 13s 234us/step - loss: 0.0145 - acc: 0.9955 - val_loss: 0.0899 - val_acc: 0.9803
/\*Epoch 10/20
/\*57000/57000 [==============================] - 14s 240us/step - loss: 0.0175 - acc: 0.9942 - val_loss: 0.1085 - val_acc: 0.9780
/\*Epoch 11/20
/\*57000/57000 [==============================] - 13s 234us/step - loss: 0.0142 - acc: 0.9954 - val_loss: 0.0811 - val_acc: 0.9853
/\*Epoch 12/20
/\*57000/57000 [==============================] - 13s 234us/step - loss: 0.0163 - acc: 0.9949 - val_loss: 0.0817 - val_acc: 0.9830
/\*Epoch 13/20
/\*57000/57000 [==============================] - 13s 235us/step - loss: 0.0114 - acc: 0.9966 - val_loss: 0.0736 - val_acc: 0.9863
/\*Epoch 14/20
/\*57000/57000 [==============================] - 13s 236us/step - loss: 0.0142 - acc: 0.9958 - val_loss: 0.0906 - val_acc: 0.9830
/\*Epoch 15/20
/\*57000/57000 [==============================] - 13s 235us/step - loss: 0.0091 - acc: 0.9970 - val_loss: 0.0952 - val_acc: 0.9803
/\*Epoch 16/20
/\*57000/57000 [==============================] - 14s 241us/step - loss: 0.0131 - acc: 0.9964 - val_loss: 0.0742 - val_acc: 0.9820
/\*Epoch 17/20
/\*57000/57000 [==============================] - 14s 237us/step - loss: 0.0088 - acc: 0.9973 - val_loss: 0.0908 - val_acc: 0.9807
/\*Epoch 18/20
/\*57000/57000 [==============================] - 13s 235us/step - loss: 0.0125 - acc: 0.9962 - val_loss: 0.0955 - val_acc: 0.9813
/\*Epoch 19/20
/\*57000/57000 [==============================] - 14s 237us/step - loss: 0.0080 - acc: 0.9975 - val_loss: 0.0935 - val_acc: 0.9840
/\*Epoch 20/20
/\*57000/57000 [==============================] - 13s 236us/step - loss: 0.0110 - acc: 0.9970 - val_loss: 0.1053 - val_acc: 0.9810
/\*10000/10000 [==============================] - 1s 78us/step
/\*the loss of test data is 0.110178467543
/\*the accuracy of test data is 0.9808


