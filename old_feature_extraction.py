import cv2 
import numpy as np
import mediapipe as mp
import torch

STANDARD_HEIGHT = 200
STANDARD_WIDTH = 200
MIN_CONFIDENCE_LEVEL = 0.7

#Extract Hand Features for inference (can be utilized for training with minor changes)
class OldExtractHandFeatures: 
    def __init__(self, raw_image):
        self.raw_image = raw_image

    def generate_features(self): 
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode = True,max_num_hands = 2,
            min_detection_confidence = MIN_CONFIDENCE_LEVEL) as hands:
            
            #For training change this line, don't need to flip (since images appear to be from back-facing camera) 
            #Convert cv2 BGR image to RGB image and flip (since image coming from front-facing camera)  
            #processed = hands.process(cv2.flip(cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB), 1))
            processed = hands.process(cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB))

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
