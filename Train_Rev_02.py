#
# Train Rev. 02
#
# Online Detection and Classification of Dynamic Hand Gestures with Recurrent 3D Convolutional Neural Networks에서
# RGB Image를 Conv 3D에 넣어서 분류하는 부분만을 적용 
#

from tabnanny import verbose
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import deque
import cv2 as cv

from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional , Dropout , BatchNormalization , Conv3D , MaxPooling3D , Flatten , ZeroPadding3D , Input





DEST_SIZE = (112 , 112)
MAX_FRAME = 128
BATCH_SIZE = 4
FRAME_SIZE = 16
EPOCHS = 5




def C3Dnet(nb_classes, input_shape):
    input_tensor = Input(shape=input_shape)
    # 1st block
    x = Conv3D(64, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv1')(input_tensor)
    x = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='valid', name='pool1')(x)
    # 2nd block
    x = Conv3D(128, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv2')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2')(x)
    # 3rd block
    x = Conv3D(256, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv3a')(x)
    x = Conv3D(256, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv3b')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3')(x)
    # 4th block
    x = Conv3D(512, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv4a')(x)
    x = Conv3D(512, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv4b')(x)
    x= MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4')(x)
    # 5th block
    x = Conv3D(512, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv5a')(x)
    x = Conv3D(512, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv5b')(x)
    x = ZeroPadding3D(padding=(0,1,1),name='zeropadding')(x)
    x= MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5')(x)
    # full connection
    x = Flatten()(x)
    x = Dense(4096, activation='relu',  name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(nb_classes, activation='softmax', name='fc8')(x)

    model = Model(input_tensor, output_tensor)
    return model




def Feature_Extractor():
    C3D_net = C3Dnet(487, (16, 112, 112, 3))
    C3D_net.load_weights("./C3D_Sport1M_weights_keras_2.2.4.h5")
    
    model = Model(C3D_net.input , C3D_net.layers[-2].output  )
    model.trainable = False
    
    return model



         






def Make_Sequential_Model():
    model = Sequential()
    model.add(Bidirectional( LSTM(1024,  return_sequences=True , activation='tanh'), input_shape=(8,4096)))
    model.add(Bidirectional( LSTM(256,  return_sequences=True , activation='tanh')))
    model.add(Bidirectional( LSTM(64,  return_sequences=False , activation='tanh')))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    return model








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








def generate_train_data( file_path , label , feature_extractor):     
       
    filename = file_path.replace("./Train_Data","C:/Users/Moon/Desktop/Hand_Gesture_Detection/Train_Data")
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
    
    output = []
    
    data = np.reshape(data, (-1 , FRAME_SIZE , 112,112,3) )

    output = feature_extractor.predict_on_batch( data )

    """
    ret = feature_extractor.predict_on_batch(data[:4, ...])
    output.append(ret)
    
    ret = feature_extractor.predict_on_batch(data[4:, ...])
    output.append(ret)
    """
   
    output = np.reshape(np.array(output) , (-1,4096))

    return output , label







#
def Train():

    # Load C3D Feature Extractor
    feature_extractor = Feature_Extractor()
    feature_extractor.summary()

    # Make RNN Model
    seq_model = Make_Sequential_Model()
    
    seq_model.summary()

    seq_model.compile(  optimizer=tf.keras.optimizers.Adam(1e-3),
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

            for batch in range(BATCH_SIZE):
                
                if idx+batch >= len(X_train):
                    break

                train_data , target = generate_train_data( X_train[ idx+batch ], y_train[ idx+batch ] , feature_extractor)
                tmp_train_data.append( train_data )
                tmp_target.append( target )

            train_data = np.reshape(tmp_train_data, (-1,8,4096))
            target = np.reshape(tmp_target, (-1,4))

            seq_model.fit(  x = train_data,
                            y = target,            
                            verbose=1)

        # Eval.
        print("### Evaluation... Epoch : {0} ###".format(epoch))
        
        if len(tmp_val_data) == 0:
            for idx in range( 0 , len(X_test)):
                train_data , target = generate_train_data( X_test[ idx ], y_test[ idx ] , feature_extractor)
                tmp_val_data.append( train_data )
                tmp_val_target.append( target )

            tmp_val_data = np.reshape(tmp_val_data, (-1,8,4096))
            tmp_val_target = np.reshape(tmp_val_target, (-1,4))

        ret = seq_model.evaluate(tmp_val_data , tmp_val_target )


    return






if __name__== "__main__":
    Train()




"""
# Layer 추가 하는 방법
a = feature_extractor.output
a = Dense(4, activation='softmax', name='fc8')(a)

feature_extractor = Model(feature_extractor.input, a)
"""