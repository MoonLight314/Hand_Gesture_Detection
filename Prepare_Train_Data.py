import numpy as np
import pandas as pd
from collections import deque
import cv2 as cv
from tqdm import tqdm

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








def generate_train_data( file_path , feature_extractor):     
       
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

    output = np.reshape(np.array(output) , (-1,4096))

    return output , filename






#
def Prepare_Train_Data():
    
    feature_extractor = Feature_Extractor()
    feature_extractor.summary()

    # Load meta data file
    meta = pd.read_csv("Meta_Data_220117_Rev_01.csv")
    # Data File Path
    file_path = meta['file_path'].tolist()

    npy_file_name =[]

    for file in tqdm(file_path):
        train_data , tmp_file_path = generate_train_data( file, feature_extractor)

        np.save( tmp_file_path.replace(".mp4","") , train_data )
        npy_file_name.append( tmp_file_path.replace(".mp4","") )

    meta['npy_path'] = npy_file_name
    meta.to_csv("Meta_Data_220126_Rev_00.csv" , index=False )
    return



if __name__== "__main__":
    Prepare_Train_Data()    