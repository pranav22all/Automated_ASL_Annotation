from matplotlib import pyplot as plt
import pyvirtualcam
from pyvirtualcam import PixelFormat
import cv2
import torch
import matplotlib

from ASL_Predictor import ASLResnet, predict_image

class ASL:
    def __init__(self, width=1280, height=720, fps=20):
        # Initialize webcam capture
        self.camera = cv2.VideoCapture(0)

        # Set width, height, and fps of camera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, fps)

        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.camera.get(cv2.CAP_PROP_FPS)

        # Text properties
        self.VERTICAL_DIST_FROM_TOP = self.height-80
        self.HORIZONTAL_DIST_FROM_RIGHT = self.width // 2
        self.FONT_FAMILY = cv2.FONT_HERSHEY_COMPLEX
        self.FONT_SIZE = 3
        self.FONT_COLOR = (255, 255, 255)
        self.FONT_STROKE = cv2.LINE_4

    def run(self):
        model = ASLResnet()
        model.load_state_dict(torch.load('checkpoints/asl-colored-resnet34-mvp.pth', map_location=torch.device('cpu')))

        with pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps, fmt=PixelFormat.BGR) as cam:
            print(f'Virtual cam started: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')
            count = 0
            while True:
                # Step 1: Read an image from the webcam
                _, frame = self.camera.read()

                if frame is None:
                    continue

                print(count)

                if count % 50 == 0:
                    img = cv2.resize(frame, (200, 200))
                    # plt.imshow(img)
                    # plt.show()
                    prediction = predict_image(img, model)
                    print(prediction)

                cv2.putText(
                    frame, # image
                    prediction, # text
                    # for longer words, starting point is further left
                    (self.HORIZONTAL_DIST_FROM_RIGHT-(40*len(prediction)), \
                        self.VERTICAL_DIST_FROM_TOP), # position at which writing has to start
                    self.FONT_FAMILY, # font family
                    self.FONT_SIZE, # font size
                    self.FONT_COLOR, # font color
                    self.FONT_STROKE) # font stroke

                # Step last_step: Send the image to the virtual camera for processing
                cam.send(frame)
                cam.sleep_until_next_frame()

                count += 1

if __name__ == '__main__':
    asl = ASL()
    asl.run()