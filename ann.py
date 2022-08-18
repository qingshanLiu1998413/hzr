import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
import numpy as np
import os
import matplotlib.pyplot as plt
#导入数据
data = pd.read_csv("2.csv")
data.drop(["Unnamed: 0"],axis=1,inplace=True)

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data();


#划分下特征和预测量

X = data.iloc[:,2:8]
y = data.iloc[:,-1]
#划分训练集和测试集

Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.1,random_state=240)
Xtrain=np.array(Xtrain)
Xtest=np.array(Xtest)
Ytrain=np.array(Ytrain)
Ytest = np.array(Ytest)

print(Xtrain)

model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse']
              )

checkpoint_save_path ="./checkpoint/ann.ckpt"

if os.path.exists(checkpoint_save_path+'.index'):
    print("--------load--------")
    model.load_weights(checkpoint_save_path)

cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                               save_weights_only=True,
                                               save_best_only=True)
history = model.fit(Xtrain,Ytrain,batch_size=16,epochs=200,
                    validation_data=(Xtest,Ytest),validation_freq=1,
                    callbacks=[cp_callback])
model.summary()
