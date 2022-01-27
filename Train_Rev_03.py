import numpy as np
import os
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional , Dropout , BatchNormalization , Conv3D , MaxPooling3D , Flatten , ZeroPadding3D , Input
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint , LearningRateScheduler



BATCH_SIZE = 4
EPOCH = 500
INIT_LEARNING_RATE = 0.05
DROP_OUT_RATE = 0.2


#
def load_npy( file_path , label ):
    filename = str(file_path.numpy())
    filename = filename.replace("b'" , "")
    filename = filename.replace("npy'" , "npy")
    data = np.load(filename)

    return data , label


#
def load_npy_dense( file_path , label ):
    filename = str(file_path.numpy())
    filename = filename.replace("b'" , "")
    filename = filename.replace("npy'" , "npy")
    data = np.load(filename)

    data = np.reshape(data ,(-1 , 8*4096))
    return data , label    





def Make_Sequential_Model():
    model = Sequential()
    model.add(Bidirectional( LSTM(2048,  return_sequences=True , activation='tanh'), input_shape=(8,4096)))
    model.add(Bidirectional( LSTM(1024,  return_sequences=True , activation='tanh')))
    model.add(Bidirectional( LSTM(256,  return_sequences=True , activation='tanh')))
    model.add(Bidirectional( LSTM(64,  return_sequences=False , activation='tanh')))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    return model







#
def Dense_Model():
    model = Sequential()

    model.add( Dense(4096, activation='relu', kernel_initializer='he_uniform' , input_shape=(None, 8 * 4096)))
    model.add( Dropout(DROP_OUT_RATE) )
    model.add( BatchNormalization() )
    """
    model.add( Dense(4096, activation='relu' , kernel_initializer='he_uniform' ))
    model.add( Dropout(DROP_OUT_RATE) )
    model.add( BatchNormalization() )
    """
    model.add( Dense(1024, activation='relu' , kernel_initializer='he_uniform' ))
    model.add( Dropout(DROP_OUT_RATE) )
    model.add( BatchNormalization() )

    model.add( Dense(256, activation='relu' , kernel_initializer='he_uniform' ))
    model.add( Dropout(DROP_OUT_RATE) )
    model.add( BatchNormalization() )

    model.add( Dense(64, activation='relu' , kernel_initializer='he_uniform' ))
    model.add( Dropout(DROP_OUT_RATE) )
    model.add( BatchNormalization() )

    model.add(Dense(4, activation='softmax'))

    return model





def lr_exp_decay(epoch, lr):
        k = 0.1
        return INIT_LEARNING_RATE * np.math.exp(-k*epoch)



def Train():

    meta = pd.read_csv("Meta_Data_220126_Rev_00.csv")

    # Data File Path
    file_path = meta['npy_path'].tolist()
    file_path = [x + ".npy" for x in file_path ]

    # Label 
    labels = meta['action'].tolist()

    print(len(file_path) , len(labels))


    le = LabelEncoder()
    le_action = le.fit(labels)
    le_action = le.transform(labels)
    print(le.classes_)

    y = tf.keras.utils.to_categorical(le_action, num_classes=4)
    print(y)


    # Make dataset
    X_train, X_test, y_train, y_test = train_test_split(file_path, y, test_size=0.25 , stratify = y)

    train_dataset = tf.data.Dataset.from_tensor_slices( (X_train , y_train) )
    val_dataset = tf.data.Dataset.from_tensor_slices( (X_test , y_test) )


    train_dataset = train_dataset.shuffle(buffer_size=len(X_train))\
                .map(   lambda X_train , y_train:
                        tf.py_function(
                            #func = load_npy,
                            func = load_npy_dense,
                            inp=[X_train , y_train],
                            Tout=[tf.float32 , tf.float32]
                        ),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        deterministic=False)\
                .batch(BATCH_SIZE)\
                .prefetch(tf.data.experimental.AUTOTUNE)     #

    
    val_dataset = val_dataset.shuffle(buffer_size=len(X_test))\
                .map(   lambda X_test , y_test:
                        tf.py_function(
                            #func = load_npy,
                            func = load_npy_dense,
                            inp=[X_test , y_test],
                            Tout=[tf.float32 , tf.float32]
                        ),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        deterministic=False)\
                .batch(BATCH_SIZE)\
                .prefetch(tf.data.experimental.AUTOTUNE)     #

    """
    # Make RNN Model
    seq_model = Make_Sequential_Model()
    
    seq_model.summary()

    seq_model.compile(  optimizer=tf.keras.optimizers.Adam(1e-3),
                        loss='categorical_crossentropy',
                        metrics=['categorical_accuracy']
    )
    """

    dense_model = Dense_Model()

    dense_model.summary()

    dense_model.compile(  optimizer=tf.keras.optimizers.Adam(1e-3),
                        loss='categorical_crossentropy',
                        metrics=['categorical_accuracy']
    )

    #
    lr_scheduler = LearningRateScheduler(lr_exp_decay, verbose=1)

    #
    log_dir = os.path.join('Logs')
    CHECKPOINT_PATH = os.path.join('CheckPoints_AgeGroup')
    tb_callback = TensorBoard(log_dir=log_dir)

    cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, 
                        monitor='val_accuracy',
                        save_best_only = True,
                        verbose = 1)


    hist = dense_model.fit(   train_dataset,
                            validation_data = val_dataset,
                            callbacks=[ lr_scheduler ],
                            epochs = EPOCH,
                            verbose = 1 
                        )


    return







if __name__== "__main__":
    Train()