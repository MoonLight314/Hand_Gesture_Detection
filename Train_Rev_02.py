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
INPUT_SHAPE = (8, 112, 112, 3)




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
def generate_data( file_path , label ):
    
    filename = str(file_path.numpy())
    filename = filename.replace("./Train_Data","C:/Users/csyi/Desktop/Hand_Gesture_Detection/Train_Data")
    #filename = 'C:/Users/csyi/Desktop/Hand_Gesture_Detection/Train_Data\Shake_Hand\\Video_Test_22.mp4'
    print(filename[0] , filename[1])

    cap = cv.VideoCapture( filename )

    if cap.isOpened():
        print("Opened")
    else:
        print("Error")
    
    data = []
    return data , label




#train_dataset = train_dataset.shuffle(buffer_size=len(X_train))\
train_dataset = train_dataset.shuffle(buffer_size=1)\
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




for data, label in train_dataset.take(1):
    print( data, label )



def define_model( input_shape ):
    
    # Create the model
    model = Sequential()
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding="same" , input_shape=input_shape))
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



model.fit( train_dataset,
            verbose=1 )