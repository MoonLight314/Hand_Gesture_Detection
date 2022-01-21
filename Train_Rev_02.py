#
# Train Rev. 02
#
# Online Detection and Classification of Dynamic Hand Gestures with Recurrent 3D Convolutional Neural Networks에서
# RGB Image를 Conv 3D에 넣어서 분류하는 부분만을 적용 
#


import numpy as np
import pandas as pd
import cv2 as cv
import os
import pydevd
from collections import deque
from tensorflow.keras.models import model_from_json

from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional , Dropout , BatchNormalization , Conv3D , MaxPooling3D , Flatten
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint



#BATCH_SIZE = 4
BATCH_SIZE = 1
DROP_OUT_RATE = 0.25

#INPUT_SHAPE = ( -1, 120, 112, 112, 3)
INPUT_SHAPE = ( -1, 32, 112, 112, 3)

DEST_SIZE = (112 , 112)
MAX_FRAME = 120


meta = pd.read_csv("Meta_Data_220117_Rev_01.csv")

# Data File Path
file_path = meta['file_path'].tolist()

# Label 
labels = meta['action'].tolist()

print(len(file_path) , len(labels))


le = LabelEncoder()
le_action = le.fit(labels)
le_action = le.transform(labels)
print(le.classes_)

y = tf.keras.utils.to_categorical(le_action, num_classes=4)
print(y)


#
X_train, X_test, y_train, y_test = train_test_split(file_path, y, test_size=0.25 , stratify = y)



train_dataset = tf.data.Dataset.from_tensor_slices( (X_train , y_train) )
val_dataset = tf.data.Dataset.from_tensor_slices( (X_test , y_test) )





#
def Adjust_Length( data ):
    #pydevd.settrace(suspend=False)
    # 목적은 ( MAX_FRAME , 112 , 112 , 3) 형태로 만드는 것
    zero = np.zeros_like( data[0] )
    d = np.array(data)

    length = d.shape[0]

    for _ in range(int((MAX_FRAME - length)/2)):
        data.append( zero )
        data.appendleft( zero )

    if len(data) != MAX_FRAME:
        data.append( zero )

    aligned_data = np.array(data)
    
    return aligned_data





#
def generate_data( file_path , label ):    
    
    #pydevd.settrace(suspend=False)
    
    filename = str(file_path.numpy())
    filename = filename.replace("b'./Train_Data","C:/Users/csyi/Desktop/Hand_Gesture_Detection/Train_Data")
    filename = filename.replace("'","")
    
    cap = cv.VideoCapture( filename )

    # 값들을 255로 나누기

    if cap.isOpened() == False:
        print("Open Error")

    data = deque([])

    while( True ):
        ret, frame = cap.read()

        if ret == False:
            break

        frame = cv.resize( frame, dsize=DEST_SIZE, interpolation=cv.INTER_AREA )
        
        data.append( frame )
    
    data = Adjust_Length( data )
    data = data / 255
    print( data.shape )

    return data , label




train_dataset = train_dataset.shuffle(buffer_size=len(X_train))\
                .map(   lambda X_train , y_train:
                        tf.py_function(
                            func = generate_data,
                            inp=[X_train , y_train],
                            Tout=[tf.float32 , tf.float32]
                        ),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        deterministic=False)\
                .batch(BATCH_SIZE)\
                .prefetch(tf.data.experimental.AUTOTUNE)     #

"""
val_dataset = val_dataset.shuffle(buffer_size=len(X_test))\
                .map( load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .batch(BATCH_SIZE)\
                .prefetch(tf.data.experimental.AUTOTUNE)    #
"""


"""
for data, label in train_dataset.take(1):
#for idx,(data, label) in enumerate(train_dataset):
    #print( idx, data, label )
    print( data, label )
    #pass
"""


def define_model( input_shape ):
    
    # Create the model
    model = Sequential()
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding="same" , input_shape=input_shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size = [1,2,2], strides = [1,2,2], padding = "same"))

    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size = [2,2,2], strides = [2,2,2], padding = "same"))
    
    model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size = [2,2,2], strides = [2,2,2], padding = "same"))
    
    model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size = [2,2,2], strides = [2,2,2], padding = "same"))

    model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(BatchNormalization())
    
    model.add( Flatten() )
    model.add( Dense(512, activation='relu') )
    model.add( Dropout(DROP_OUT_RATE) )
    model.add( Dense(512, activation='relu') )
    model.add( Dropout(DROP_OUT_RATE) )

    return model



model = define_model( INPUT_SHAPE )
print( model.summary() )

"""
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)


model.fit( train_dataset,
            verbose=1 )
"""            