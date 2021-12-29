import cv2
import numpy as np
import os
import time
import mediapipe as mp



# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Train_Data') 

# Actions that we try to detect
actions = np.array(['jamjam' , 'hi' , 'go_away'])

# Thirty videos worth of data
no_sequences = 10

# Videos are going to be 30 frames in length
DATA_FRAMES_PER_DATA = 60

# Folder start
start_folder = 0

# Frame Delay Time
FRAME_DELAY_TIME = 5




mp_hands = mp.solutions.hands # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles



def draw_landmark( image , hands ):
    # Make detections
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = hands.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
                
    return image , results





def GetLastDataSetNum():
    
    path = os.path.join('Train_Data') 

    last_num = os.listdir( path )

    return len(last_num)








def GatherTrainData():   

    cap = cv2.VideoCapture(0)
    
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    

    # Set mediapipe model 
    with mp_hands.Hands( max_num_hands = 1,
                        min_detection_confidence=0.5, 
                        min_tracking_confidence=0.5) as hands:
        key = 0

        while cap.isOpened() and key != 27:

            last_num = GetLastDataSetNum()
            file_name = os.path.join('Train_Data') 
            file_name = file_name + "\Video_Test_" + str(last_num) + ".mp4"
            out = cv2.VideoWriter(file_name, fourcc, 24, (w, h))
            is_recording = False

            while( True ):
                # Read feed
                success, image = cap.read()

                key = cv2.waitKey(1)
                if key == 32:
                    if is_recording:
                        is_recording = False
                        break
                    else:
                        is_recording = True

                if is_recording:
                    #cv2.putText(image, 'Recording...', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    pass
                else:
                    cv2.putText(image, 'Press Space to start recording', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)

                
                # Break gracefully
                key = cv2.waitKey(1)
                if key == 27:
                    break

                # Land Mark만 동영상으로 저장해 놓으면 나중에 사용할 일이 있을까?
                #image ,results = draw_landmark( image , hands )

                cv2.imshow('OpenCV Feed', image)

                if is_recording:
                    out.write( image )

            
            out.release()

        cap.release()        
        cv2.destroyAllWindows()







if __name__== '__main__':
    GatherTrainData()



'''
if results.multi_hand_landmarks == None:
    continue                
    

# Feature 추출
for hand_landmarks in results.multi_hand_landmarks:
    for landmark in hand_landmarks.landmark:
        #test = np.array([landmark.x, landmark.y, landmark.z])
        landmark_feature.append(landmark.x)
        landmark_feature.append(landmark.y)
        landmark_feature.append(landmark.z)

cv2.imshow('OpenCV Feed', image)                    
cv2.waitKey(FRAME_DELAY_TIME)
frame_num = frame_num + 1

landmark_data.append( landmark_feature )

if frame_num == DATA_FRAMES_PER_DATA:
    n = np.array( landmark_data )
    npy_path = os.path.join(DATA_PATH, action, str(last_num))
    #print( np.array(landmark_feature).flatten().shape )
    np.save(npy_path, n)
    last_num = last_num + 1
    break
'''