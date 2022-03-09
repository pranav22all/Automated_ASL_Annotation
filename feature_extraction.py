import cv2 
import numpy as np
import mediapipe as mp
import torch

STANDARD_HEIGHT = 200
STANDARD_WIDTH = 200
MIN_CONFIDENCE_LEVEL = 0.7

#Extract Hand Features for inference (can be utilized for training with minor changes)
class ExtractHandFeatures: 
    def __init__(self, raw_image):
        self.raw_image = raw_image

    def generate_features(self): 
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode = True,max_num_hands = 2,
            min_detection_confidence = MIN_CONFIDENCE_LEVEL) as hands:
            
            #For training change this line, don't need to flip (since images appear to be from back-facing camera) 
            #Convert cv2 BGR image to RGB image and flip (since image coming from front-facing camera)  
            processed = hands.process(cv2.flip(cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB), 1))
            # processed = hands.process(cv2.flip(self.raw_image, 1))
            # processed = hands.process(cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)) 

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

class ExtractAngleFeatures: 
    def __init__(self, raw_image):
        self.raw_image = raw_image

    def get_angle_between_vectors(self, u: np.ndarray, v: np.ndarray) -> float:
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        return np.arccos(dot_product / norm)

    def generate_features(self): 
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode = True,max_num_hands = 2,
            min_detection_confidence = MIN_CONFIDENCE_LEVEL) as hands:
            processed = hands.process(cv2.flip(cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB), 1))

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
                        angle = self.get_angle_between_vectors(connection_from, connection_to)
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
                        angle = self.get_angle_between_vectors(connection_from, connection_to)
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
        