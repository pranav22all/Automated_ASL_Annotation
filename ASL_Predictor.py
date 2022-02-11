import random

class ASL_Predictor:
    def __init__(self):
        self.classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
                'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                'u', 'v', 'w', 'x', 'y', 'z', 'SPACE', 'DEL']

        self.num_classes = len(self.classes)

    def predict(self, frame):
        # Returns a random letter
        return self.classes[random.randint(0, self.num_classes-1)]