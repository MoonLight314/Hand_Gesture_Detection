#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os

import tensorflow_hub as hub
import tensorflow as tf


# In[2]:


def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            #frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
        
    return np.array(frames) / 255.0


# In[3]:


video_file_path = os.path.join('Train_Data','1_Finger_Click','Video_Test_4 (2).mp4')


# In[4]:


sample_video = load_video( video_file_path )


# In[5]:


sample_video.shape


# In[ ]:





# In[18]:


i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]


# In[21]:


model_input


# In[24]:


logits = i3d(model_input)['default'][0]


# In[ ]:





# In[29]:


from urllib import request  # requires python3


# In[30]:


# Get the kinetics-400 action labels from the GitHub repository.
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
with request.urlopen(KINETICS_URL) as obj:
    labels = [line.decode("utf-8").strip() for line in obj.readlines()]
print("Found %d labels." % len(labels))


# In[ ]:





# In[31]:


probabilities = tf.nn.softmax(logits)


# In[32]:


print("Top 5 actions:")
for i in np.argsort(probabilities)[::-1][:5]:
    print(f"  {labels[i]:22}: {probabilities[i] * 100:5.2f}%")


# In[ ]:




