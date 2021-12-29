import numpy as np
import os
import mediapipe as mp
import cv2
import time

from keras.models import load_model, save_model

#from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
#from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint


mp_hands = mp.solutions.hands # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles

# Actions that we try to detect
actions = np.array(['jamjam' , 'hi' , 'go_away'])

CHECKPOINT_PATH = os.path.join('CheckPoints')

# Videos are going to be 30 frames in length
DATA_FRAMES_PER_DATA = 60





def draw_landmark(image, hands):
    # Make detections
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = hands.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    return image, results






def Inference():

    model = load_model("CheckPoints")
    cap = cv2.VideoCapture(0)

    # Set mediapipe model
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:

        if cap.isOpened():

            landmark_feature = []
            feature_data = []

            while (True):
                # Read feed
                success, image = cap.read()

                image, results = draw_landmark(image, hands)                

                # Break gracefully
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break

                if results.multi_hand_landmarks == None:
                    cv2.putText(image, "Not Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
                    cv2.imshow('Result', image)
                    continue

                # Feature 추출
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmark_feature.append(landmark.x)
                        landmark_feature.append(landmark.y)
                        landmark_feature.append(landmark.z)

                feature_data.append( landmark_feature )
                feature_data = feature_data[-60:]
                landmark_feature = []

                if len(feature_data) == DATA_FRAMES_PER_DATA:
                    data = np.array( feature_data ).reshape(-1 , DATA_FRAMES_PER_DATA, 63)
                    
                    pred = model.predict( data )                                        
                    print( actions[ np.argmax(pred) ] , np.max( pred ) ) # Predict 시간은 보통 60~70ms
                    cv2.putText(image, "Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
                    feature_data = []
                
                # Show to screen
                cv2.imshow('Result', image)


    cap.release()
    cv2.destroyAllWindows()

    return





if __name__== '__main__':
    Inference()