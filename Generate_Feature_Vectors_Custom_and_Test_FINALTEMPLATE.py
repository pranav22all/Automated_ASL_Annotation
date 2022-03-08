#!/usr/bin/env python
# coding: utf-8

# In[8]:


import mediapipe as mp
import numpy as np
import cv2 as cv2
import os
import math
import numpy as np
import torch
from tqdm import tqdm 


# In[9]:


MIN_CONFIDENCE_LEVEL = 0.7
folders = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


# In[10]:


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# In[11]:


#np.load("train_numpy_arrays/asl_alphabet_train/asl_alphabet_train/B/B1727.npy")


# In[12]:


#Custom Dataset: /home/pranav/gcloud/Custom_Alphabet_Dataset
#Full Combined Dataset: /home/pranav/gcloud/DatasetsCombinedFULL 
#Sampling Combined Dataset: /home/pranav/gcloud/DatasetsCombinedSAMPLED
#Test Dataset: /home/pranav/gcloud/ASL_Alphabet_Test
#"OLD', 'HANDED', 'ANGLE'
RUN_TYPE = 'OLD'
train_dir = '/home/pranav/gcloud/Custom_Alphabet_Dataset'
test_dir = '/home/pranav/gcloud/ASL_Alphabet_Test'
train_output_dir = 'train_numpy_arrays/' + RUN_TYPE + '/Custom_Alphabet_Dataset'
test_output_dir = 'test_numpy_arrays/' + RUN_TYPE + '/ASL_Alphabet_Test'

print(RUN_TYPE)
print("Train dir is:")
print(train_dir)
print("Test dir is:")
print(test_dir)
print("Train output dir is:")
print(train_output_dir)
print("Test output dir is:")
print(test_output_dir)


# In[ ]:


#Create output directories for train  and test: 
for curr_folder in folders: 
    path = train_output_dir + '/' + curr_folder
    if not(os.path.exists(path)):
        os.makedirs(path)
    
for curr_folder in folders: 
    path = test_output_dir + '/' + curr_folder
    if not(os.path.exists(path)):
        os.makedirs(path)


# In[13]:


#Get number of files in each directory / 
for letter_dir in os.listdir(train_dir): 
    letter_path = os.path.join(train_dir, letter_dir)
    num_files = len(os.listdir(letter_path))
    # print("Currently looking at the following folder: " + letter_dir)
    # print("There are " + str(num_files) + " files in this folder")


# In[ ]:


def get_old_features(processed): 
    #No hand detected (Figure out how we want to handle, 126 vector with all 0s?): 
    if not processed.multi_hand_landmarks: 
        zeros = torch.tensor(np.array([0] * 126), dtype=torch.float32)
        return zeros

    feature_vector = [] 
    #Could have one or two hands: 
    for hand in processed.multi_hand_landmarks: 
        for curr_landmark in hand.landmark: 
            x = curr_landmark.x 
            feature_vector.append(x)

            y = curr_landmark.y 
            feature_vector.append(y)

            z = curr_landmark.z
            feature_vector.append(z)

    #If we have just one hand, zero out the remaining (to ensure constant vector size of 126)
    #Might cause problems in one-hand case if we care which hand is visible/showing sign language
    #Solution to this is to use processed.multi_handedness
    if (len(feature_vector) == 63):
        zero_vector = [0] * 63 
        feature_vector.extend(zero_vector)

    output = torch.tensor(np.array(feature_vector), dtype=torch.float32)
    return output


# In[ ]:


def get_handed_features(processed): 
    #No hand detected (Figure out how we want to handle, 126 vector with all 0s?): 
    if not processed.multi_hand_landmarks: 
        zeros = torch.tensor(np.array([0] * 126), dtype=torch.float32)
        return zeros

    feature_vector = []         
    hands = [] 

    for idx, hand_handedness in enumerate(processed.multi_handedness):
        hands.append(hand_handedness.classification[0].label)


    #Left hand is first 63, Right hand is last 63
    #LEFT HAND ONLY CASE: 
    if (len(hands) == 1 and hands[0] == "Left"):
        for hand in processed.multi_hand_landmarks: 
            for curr_landmark in hand.landmark: 
                x = curr_landmark.x 
                feature_vector.append(x)

                y = curr_landmark.y 
                feature_vector.append(y)

                z = curr_landmark.z
                feature_vector.append(z)
        zero_vector = [0] * 63 
        feature_vector.extend(zero_vector)

    #RIGHT HAND ONLY CASE: 
    if (len(hands) == 1 and hands[0] == "Right"):
        # print("Detected only right hand")
        for hand in processed.multi_hand_landmarks: 
            for curr_landmark in hand.landmark: 
                x = curr_landmark.x 
                feature_vector.append(x)

                y = curr_landmark.y 
                feature_vector.append(y)

                z = curr_landmark.z
                feature_vector.append(z)
        zero_vector = [0] * 63 
        feature_vector = zero_vector + feature_vector

    #BOTH HANDS CASE: 
    if (len(hands) == 2):
        # print("Detected both hands")
        zeros = torch.tensor(np.array([0] * 126), dtype=torch.float32)
        return zeros

    output = torch.tensor(np.array(feature_vector), dtype=torch.float32)
    #print(output)
    return output


# In[ ]:


def get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
    dot_product = np.dot(u, v)
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    return np.arccos(dot_product / norm)

def get_angle_features(processed): 
    #No hand detected (Figure out how we want to handle, 882 vector with all 0s?): 
    if not processed.multi_hand_landmarks: 
        #print("no hand")
        zeros = torch.tensor(np.array([0] * 882), dtype=torch.float32)
        return zeros

    angles_list = []         
    hands = [] 

    for idx, hand_handedness in enumerate(processed.multi_handedness):
        hands.append(hand_handedness.classification[0].label)

    #LEFT HAND ONLY CASE: 
    if (len(hands) == 1 and hands[0] == "Left"):
        landmarks = np.zeros((21, 3))
        index = 0 
        #print("Detected only left hand")
        for hand in processed.multi_hand_landmarks:   
            for curr_landmark in hand.landmark: 
                x = curr_landmark.x 
                y = curr_landmark.y 
                z = curr_landmark.z
                landmarks[index] = [x, y, z]
                index += 1

        # print("Landmarks is:")
        # print(landmarks)

        connections = mp_hands.HAND_CONNECTIONS
        # print(connections)
        # print(len(connections))

        difference_connect_vector = list(map(lambda t: landmarks[t[1]] - landmarks[t[0]], connections))
        # print(difference_connect_vector)
        # print(len(difference_connect_vector))

        for connection_from in difference_connect_vector:
            for connection_to in difference_connect_vector:
                angle = get_angle_between_vectors(connection_from, connection_to)
                # If the angle is not null we store it else we store 0
                if angle == angle:
                    angles_list.append(angle)
                else:
                    angles_list.append(0)
        # print("Angles list is:")
        # print(angles_list)
        # print(len(angles_list))
        zero_vector = [0] * 441 
        angles_list.extend(zero_vector)

    #RIGHT HAND ONLY CASE: 
    if (len(hands) == 1 and hands[0] == "Right"):
        landmarks = np.zeros((21, 3))
        index = 0 
        #print("Detected only right hand")
        for hand in processed.multi_hand_landmarks:   
            for curr_landmark in hand.landmark: 
                x = curr_landmark.x 
                y = curr_landmark.y 
                z = curr_landmark.z
                landmarks[index] = [x, y, z]
                index += 1

        # print("Landmarks is:")
        # print(landmarks)
        connections = mp_hands.HAND_CONNECTIONS
        # print(connections)
        # print(len(connections))

        difference_connect_vector = list(map(lambda t: landmarks[t[1]] - landmarks[t[0]], connections))
        # print(difference_connect_vector)
        # print(len(difference_connect_vector))

        for connection_from in difference_connect_vector:
            for connection_to in difference_connect_vector:
                angle = get_angle_between_vectors(connection_from, connection_to)
                # If the angle is not null we store it else we store 0
                if angle == angle:
                    angles_list.append(angle)
                else:
                    angles_list.append(0)
        zero_vector = [0] * 441 
        angles_list = zero_vector + angles_list

    #BOTH HANDS CASE: 
    if (len(hands) == 2):
        #print("Detected both hands")
        zeros = torch.tensor(np.array([0] * 882), dtype=torch.float32)
        return zeros

    output = torch.tensor(np.array(angles_list), dtype=torch.float32)
    return output


# In[14]:


print("Currently processing the train directory images")
for letter_dir in os.listdir(train_dir): 
    print("Currently processing the following folder: " + letter_dir)
    letter_path = os.path.join(train_dir, letter_dir)
    num_files = len(os.listdir(letter_path))
    print("There are " + str(num_files) + " files in this folder")
    
    for file_name in tqdm(os.listdir(letter_path)): 
        if file_name.endswith(".jpg"):
            f_name, f_ext = os.path.splitext(file_name)

            image_path = os.path.join(letter_path, file_name)
            image = cv2.imread(image_path)

            output_path = os.path.join(train_output_dir, letter_dir, f_name)
            with mp_hands.Hands(static_image_mode = True,max_num_hands = 2,
                min_detection_confidence = MIN_CONFIDENCE_LEVEL) as hands:

                processed = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
                if RUN_TYPE == "OLD": 
                    output = get_old_features(processed)
                if RUN_TYPE == "HANDED": 
                    output = get_handed_features(processed)
                if RUN_TYPE == "ANGLE": 
                    output = get_angle_features(processed)
                np.save(output_path, output)
    
    output_folder_path = os.path.join(train_output_dir, letter_dir)
    num_generated_files = len(os.listdir(output_folder_path))
    print("Finished with folder")
    print("Generated " + str(num_generated_files) + " npy files")
                    
            


# In[15]:


#Hasn't been tested yet: 
print("Currently processing the test directory images")
for letter_dir in os.listdir(test_dir): 
    print("Currently processing the following folder: " + letter_dir)
    letter_path = os.path.join(test_dir, letter_dir)
    num_files = len(os.listdir(letter_path))
    print("There are " + str(num_files) + " files in this folder")
    
    for file_name in tqdm(os.listdir(letter_path)): 
        if file_name.endswith(".jpg"):
            f_name, f_ext = os.path.splitext(file_name)

            image_path = os.path.join(letter_path, file_name)
            image = cv2.imread(image_path)

            output_path = os.path.join(test_output_dir, letter_dir, f_name)
            with mp_hands.Hands(static_image_mode = True,max_num_hands = 2,
                min_detection_confidence = MIN_CONFIDENCE_LEVEL) as hands:

                processed = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
                if RUN_TYPE == "OLD": 
                    output = get_old_features(processed)
                if RUN_TYPE == "HANDED": 
                    output = get_handed_features(processed)
                if RUN_TYPE == "ANGLE": 
                    output = get_angle_features(processed)
                np.save(output_path, output)
    
    output_folder_path = os.path.join(test_output_dir, letter_dir)
    num_generated_files = len(os.listdir(output_folder_path))
    print("Finished with folder")
    print("Generated " + str(num_generated_files) + " npy files")

######
                 
            

