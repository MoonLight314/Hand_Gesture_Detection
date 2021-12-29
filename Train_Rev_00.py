import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint



# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Train_Data') 

# Actions that we try to detect
actions = np.array(['jamjam' , 'hi' , 'go_away'])

# Videos are going to be 30 frames in length
DATA_FRAMES_PER_DATA = 60

label_map = {label:num for num, label in enumerate(actions)}

log_dir = os.path.join('Logs')
CHECKPOINT_PATH = os.path.join('CheckPoints')
tb_callback = TensorBoard(log_dir=log_dir)

cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor='val_categorical_accuracy',
                     save_best_only = True,
                     verbose = 1)




def PreProcess():
    sequences, labels = [], []

    for action in actions:
        n = os.listdir(os.path.join('Train_Data' , action))
        
        for data_file_name in os.listdir(os.path.join('Train_Data' , action)):
            res = np.load(os.path.join(DATA_PATH, action, data_file_name))
            sequences.append(res)
            labels.append(label_map[action])

    y = to_categorical(labels).astype(int)
        
    return np.array(sequences) , np.array(y)







def Define_Model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(DATA_FRAMES_PER_DATA,63)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    return model



# Simple하게 만든 Version
def Define_Model_Rev_01():
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(DATA_FRAMES_PER_DATA,63)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    return model








def Train(X , Y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25 , stratify = Y)

    model = Define_Model_Rev_01()
    
    model.compile(  optimizer='Adam', 
                    loss='categorical_crossentropy', 
                    metrics=['categorical_accuracy'])

    model.fit(  X_train, y_train, 
                validation_data=(X_test , y_test),
                epochs=2000, 
                callbacks=[cp , tb_callback])

    return



if __name__== '__main__':
    X,Y = PreProcess()
    Train( X , Y )