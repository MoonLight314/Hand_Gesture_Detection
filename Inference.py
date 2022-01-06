import numpy as np
import os
import mediapipe as mp
import cv2
from collections import deque

from keras.models import load_model, save_model




mp_hands = mp.solutions.hands # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles



# Actions that we try to detect
actions = np.array(['1_Finger_Click', '2_Fingers_Left', '2_Fingers_Right','Shake_Hand'])



CHECKPOINT_PATH = os.path.join('CheckPoints_Hand_Gesture_Ver_00_1.00')



DATA_FRAMES_PER_DATA = 115
MIN_FRAMES_PER_DATA = 30
MIN_NO_HAND_FRAMES_COUNT = 4





def Get_Hand_Land_Mark( image , hands ):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = hands.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR

    return results







def Extract_Hand_Land_Mark( results ):
    one_frame_result = []
                    
    for hand_landmarks in results.multi_hand_landmarks:
        for ids, landmrk in enumerate(hand_landmarks.landmark):
            one_frame_result.append( landmrk.x )
            one_frame_result.append( landmrk.y )
            one_frame_result.append( landmrk.z )        

    return one_frame_result





def Make_Aligned_Data( data ):    
    '''
    Predict에 넣을 수 있도록 DATA_FRAMES_PER_DATA 길이에 딱 맞게 맞춘 Data를 만든다.
    '''
    
    zero_list = [0] * 63

    QL = len(data)

    if QL == DATA_FRAMES_PER_DATA:
        aligned_data = data

    # 길이가 너무 길면 앞/뒤로 짜른다.
    elif QL > DATA_FRAMES_PER_DATA:
        
        for cnt in range(int((QL-DATA_FRAMES_PER_DATA)/2)):
            data.pop()
            data.popleft()

        if len(data) != DATA_FRAMES_PER_DATA:
            data.pop()

        aligned_data = data

    # 길이가 짧으면 앞/뒤로 Zero Padding한다.
    else:
        for cnt in range(int((DATA_FRAMES_PER_DATA - QL)/2)):
            data.append( zero_list )
            data.appendleft( zero_list )

        if len(data) != DATA_FRAMES_PER_DATA:
            data.append( zero_list )

        aligned_data = data

    return aligned_data





def Inference():
    zero_list = [0] * 63

    model = load_model( CHECKPOINT_PATH )
    cap = cv2.VideoCapture(0)

    # Set mediapipe model
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:

        if cap.isOpened():

            tmp_result = deque([])
            countinuous_no_hand_frame_count = 0

            while (True):
                
                success, image = cap.read()                

                results = Get_Hand_Land_Mark( image , hands )

                # Land Mark를 Detect했으면... 즉, Hand가 있으면
                if results.multi_hand_landmarks:
                    hand_land_mark = Extract_Hand_Land_Mark( results )
                    tmp_result.append( hand_land_mark )
                    
                    cv2.putText(image, "Hand Detected", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)

                    countinuous_no_hand_frame_count = 0
                

                # Hand를 못 찾았으면
                else:

                    if countinuous_no_hand_frame_count > MIN_NO_HAND_FRAMES_COUNT:

                        countinuous_no_hand_frame_count = 0
                    
                        if len(tmp_result) >= MIN_FRAMES_PER_DATA:
                            data = Make_Aligned_Data( tmp_result )
                            data = np.array(tmp_result).reshape(-1 , DATA_FRAMES_PER_DATA, 63)                    
                            pred = model.predict( data )
                            print( actions[ np.argmax(pred) ] , np.max( pred ) )
                            tmp_result.clear()

                        else:
                            #print("Frames are too short." , len(tmp_result))
                            tmp_result.clear()
                    
                    else:
                        countinuous_no_hand_frame_count = countinuous_no_hand_frame_count + 1

                cv2.imshow('Result', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    cap.release()
    cv2.destroyAllWindows()

    return





if __name__== '__main__':
    Inference()



"""
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
"""    