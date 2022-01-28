import pandas as pd
import cv2 as cv
import numpy as np
from collections import deque

from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional , Dropout , BatchNormalization , GlobalAveragePooling2D



MAX_FRAME = 128
BATCH_SIZE = 32
EPOCHS = 5
DEST_SIZE = (160,160)




def Feature_Extractor():
    model = Sequential()

    model.add( MobileNetV2(include_top = False, 
                        input_shape=(160, 160, 3)#,
                        #weights = None 
                        ))   

    model.add(GlobalAveragePooling2D()) 
    model.trainable = False

    return model







def Make_RNN_Model():
    model = Sequential()
    model.add(Bidirectional( LSTM(512,  return_sequences=True , activation='tanh'), input_shape=(MAX_FRAME , 1280)))
    model.add(Bidirectional( LSTM(256,  return_sequences=True , activation='tanh')))
    model.add(Bidirectional( LSTM(64,  return_sequences=False , activation='tanh')))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    return model




def Adjust_Length( data ):
    #pydevd.settrace(suspend=False)   
    
    front = data[0]
    back = data[-1]

    d = np.array(data)
    data = deque(data)

    length = d.shape[0]

    for _ in range(int((MAX_FRAME - length)/2)):
        data.append( back )
        data.appendleft( front )

    if len(data) != MAX_FRAME:
        data.append( back )

    aligned_data = np.array(data)
    
    return aligned_data





def generate_train_data( file_path , label , feature_extractor):     

    output = []

    for f in file_path:
        
        batch_frames = []

        filename = f.replace("./Train_Data","C:/Users/Moon/Desktop/Hand_Gesture_Detection/Train_Data")
        filename = filename.replace("'","")   

        cap = cv.VideoCapture( filename )

        if cap.isOpened() == False:
            print("Open Error")

        data = deque([])

        while( True ):
            ret, frame = cap.read()

            if ret == False:
                break

            frame = cv.resize( frame, dsize=DEST_SIZE, interpolation=cv.INTER_AREA )
            frame = tf.keras.applications.mobilenet_v2.preprocess_input( frame )
            batch_frames.append( frame )
        
        batch_frames = np.reshape( np.array(batch_frames) , (-1,160,160,3))

        ret = feature_extractor.predict_on_batch( batch_frames )

        data = Adjust_Length( ret )
        output.append(data)
        
    output = np.reshape(np.array(output) , (-1,128,1280))
    label = np.reshape(np.array(label) , (-1,4))

    return output , label










#
def Train():

    # Load Feature Extractor
    feature_extractor = Feature_Extractor()
    feature_extractor.summary()

    RNN_Model = Make_RNN_Model()

    RNN_Model.compile(  optimizer=tf.keras.optimizers.Adam(1e-3),
                        loss='categorical_crossentropy',
                        metrics=['categorical_accuracy']
    )   


    # Load meta data file
    meta = pd.read_csv("Meta_Data_220117_Rev_01.csv")
    # Data File Path
    file_path = meta['file_path'].tolist()
    # Label 
    labels = meta['action'].tolist()
    #print(len(file_path) , len(labels))

    le = LabelEncoder()
    le_action = le.fit(labels)
    le_action = le.transform(labels)
    print(le.classes_)

    y = tf.keras.utils.to_categorical(le_action, num_classes=4)
    print(y)

    # Train / Test 나누기
    X_train, X_test, y_train, y_test = train_test_split(file_path, y, test_size=0.25 , stratify = y)

    tmp_val_data = []
    tmp_val_target = []

    #
    for epoch in range(EPOCHS):

        print("### Epoch : {0} ###\n\n".format(epoch))

        for idx in range( 0 , len(X_train) , BATCH_SIZE):

            tmp_train_data = []
            tmp_target = []

            batch_file_list = []
            batch_target = []

            for batch in range(BATCH_SIZE):                

                if idx+batch >= len(X_train):
                    break
                
                batch_file_list.append( X_train[idx+batch] )
                batch_target.append( y_train[idx+batch] )

            train_data , target = generate_train_data( batch_file_list, batch_target , feature_extractor)

            RNN_Model.fit(  x = train_data,
                            y = target,            
                            verbose=1)

        # Eval.
        print("### Evaluation... Epoch : {0} ###".format(epoch))
        
        if len(tmp_val_data) == 0:
            tmp_val_data , tmp_val_target = generate_train_data( X_test , y_test , feature_extractor)

        ret = RNN_Model.evaluate(tmp_val_data , tmp_val_target )

    return





if __name__== "__main__":
    Train()