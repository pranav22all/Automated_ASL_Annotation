import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import mediapipe as mp
import numpy as np
import cv2 as cv2
from torchvision.datasets import ImageFolder
from feature_extraction import ExtractHandFeatures
from feature_extraction import ExtractAngleFeatures
from old_feature_extraction import OldExtractHandFeatures

STANDARD_HEIGHT = 200
STANDARD_WIDTH = 200
MIN_CONFIDENCE_LEVEL = 0.7

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

device = get_default_device()

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
                'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                'u', 'v', 'w', 'x', 'y', 'z', 'SPACE', 'DEL']

train_dir = '../asl/asl_alphabet_train/asl_alphabet_train'
# train_dir = '/Users/pranav/ASL_ALPHABET_DATASET/asl_alphabet_train/asl_alphabet_train'

class ASL_Predictor:
    def __init__(self):
        self.classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
                'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                'u', 'v', 'w', 'x', 'y', 'z', 'SPACE', 'DEL']

        self.num_classes = len(self.classes)

    def predict(self, frame):
        # Returns a random letter
        return self.classes[random.randint(0, self.num_classes-1)]


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

class ASLResnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 29)
        self.dataset = ImageFolder(train_dir)

    
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True

class ASLMediaPipeNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(126, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 29),
        )
        
        self.network = self.linear_relu_stack
        self.dataset = ImageFolder(train_dir)
    
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
#         for param in self.network.fc.parameters():
#             param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True

class ASLDeepNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(126, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.25),
            nn.Linear(128, 29),           
        )
        
        self.network = self.linear_relu_stack
        self.dataset = ImageFolder(train_dir)
    
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
#         for param in self.network.fc.parameters():
#             param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True

# def process_mediapipe(img):
#     image = np.array(img)
#     mp_hands = mp.solutions.hands
#     with mp_hands.Hands(static_image_mode = True,max_num_hands = 2,
#         min_detection_confidence = MIN_CONFIDENCE_LEVEL) as hands:

#         #For training change this line, don't need to flip (since images appear to be from back-facing camera) 
#         #Convert cv2 BGR image to RGB image and flip (since image coming from front-facing camera)  
#         # processed = hands.process(cv2.flip(image, 1))
#         processed = hands.process(image)

#         #No hand detected (Figure out how we want to handle, 126 vector with all 0s?): 
#         if not processed.multi_hand_landmarks: 
#             zeros = torch.tensor(np.array([0] * 126), dtype=torch.float32)
#             return zeros

#         feature_vector = [] 
#         #Could have one or two hands: 
#         for hand in processed.multi_hand_landmarks: 
#             for curr_landmark in hand.landmark: 
#                 x = curr_landmark.x 
#                 feature_vector.append(x)

#                 y = curr_landmark.y 
#                 feature_vector.append(y)

#                 z = curr_landmark.z
#                 feature_vector.append(z)

#         #If we have just one hand, zero out the remaining (to ensure constant vector size of 126)
#         #Might cause problems in one-hand case if we care which hand is visible/showing sign language
#         #Solution to this is to use processed.multi_handedness
#         if (len(feature_vector) == 63):
#             zero_vector = [0] * 63 
#             feature_vector.extend(zero_vector)
        
#         output = torch.tensor(np.array(feature_vector), dtype=torch.float32)

#         return output

dataset = ImageFolder(train_dir)
    
def predict_image(img, model, mediapipe=False, feature='OLD'):
    # Convert to a batch of 1
    if mediapipe:
        features_obj = None
        if feature == 'ANGLED':
            features_obj = ExtractAngleFeatures(img)

        elif feature == 'HANDED':
            features_obj = ExtractHandFeatures(img)

        else:
            features_obj = OldExtractHandFeatures(img)

        curr_features = features_obj.generate_features()
        xb = to_device(curr_features.unsqueeze(0), device)
    else:
        img = torch.tensor(img, dtype=torch.float).reshape((3, img.shape[1], img.shape[0]))
        xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)

    return dataset.classes[preds[0].item()]