from matplotlib import pyplot as plt
import pyvirtualcam
from pyvirtualcam import PixelFormat
import cv2
import torch
import pyttsx3

from ASL_Predictor import ASLMediaPipeNet, ASLResnet, ASLDeepNet, predict_image


class ASL:
    def __init__(self, width=1280, height=720, fps=20):
        # Initialize webcam capture
        self.camera = cv2.VideoCapture(1)

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
        self.FONT_FAMILY = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SIZE = 2
        self.FONT_COLOR = (255, 255, 255)
        self.FONT_STROKE = cv2.LINE_4

        self.engine = pyttsx3.init()

        self.LETTERS = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                        'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                        'W', 'X', 'Y', 'Z'}

    def run(self):
        ######## ASLResnet ########
        # model = ASLResnet()
        # model.load_state_dict(torch.load('checkpoints/asl-colored-resnet34-mvp.pth', map_location=torch.device('cpu')))

        ######## ASLMediaPipeNet ########
        # model = ASLMediaPipeNet()
        # model.load_state_dict(torch.load('checkpoints/asl-colored-mediapipe-mvp.pth', map_location=torch.device('cpu')))
        # model.load_state_dict(torch.load('checkpoints/asl-colored-mediapipe-mvp2.pth', map_location=torch.device('cpu')))

        ######## ASLDeepNet ########
        model = ASLDeepNet()
        model.load_state_dict(torch.load('checkpoints/asl-colored-mediapipe-mvp2-customdata-handed-combined.pth', map_location=torch.device('cpu')))
        model.eval()

        self.engine.startLoop(False)
        with pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps, fmt=PixelFormat.BGR) as cam:
            print(f'Virtual cam started: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')
            count = 0
            buffer = ''
            candidate_votes = list()
            space_count = 0

            while True:
                # Step 1: Read an image from the webcam
                _, frame = self.camera.read()

                if frame is None:
                    continue

                print(count)

                prediction = predict_image(
                    frame, model, mediapipe=True, feature='HANDED')

                print(prediction)
                candidate_votes.append(prediction)

                if count > 0 and count % 20 == 0:
                    # "vote" on the prediction that is most frequent
                    best_pred = max(set(candidate_votes),
                                    key=candidate_votes.count)

                    if len(buffer) >= 16:
                        space_index = buffer.find(' ')
                        if space_index == -1:
                            buffer = ''

                        else:
                            buffer = buffer[space_index+1:]

                    # only add prediction to buffer if it's a letter (ignoring SPACE, DELETE, NOTHING)
                    if best_pred in self.LETTERS:
                        buffer += best_pred

                    # only add space if we have at least a non-space character in the buffer
                    elif buffer and buffer[-1] != ' ' and (best_pred == 'space' or best_pred == 'nothing'):
                        word = buffer.split(' ')[-1]
                        self.engine.say(word.lower())
                        self.engine.iterate()
                        buffer += ' '

                    if buffer and best_pred == 'del':
                        buffer = buffer[:-1]

                    #     # "vote" on the prediction that is most frequent
                    #     best_pred = max(set(candidate_votes),
                    #                     key=candidate_votes.count)

                    #     spoken = False


                    #     if len(buffer) >= 16:
                    #         space_index = buffer.find(' ')
                    #         if space_index == -1:
                    #             buffer = ''

                    #         else:
                    #             buffer = buffer[space_index+1:]

                    #     if best_pred == 'space' or best_pred == 'nothing':
                    #         space_count += 1

                    #     # only add prediction to buffer if it's a letter (ignoring SPACE, DELETE, NOTHING)
                    #     if best_pred in self.LETTERS:
                    #         buffer += best_pred
                    #         space_count = 0

                    #     print('space count', space_count)
                    #     if space_count >= 2:
                    #         if buffer and buffer[-1] == ' ':
                    #             buffer = buffer[:-1]

                    #         spoken = True

                    #         print('SAYING THIS:', buffer.lower() + '.')
                    #         self.engine.say(buffer.lower())
                    #         self.engine.iterate()
                    #         space_count = 0

                    #     # only add space if we have at least a non-space character in the buffer
                    #     if buffer and buffer[-1] != ' ' and (best_pred == 'space' or best_pred == 'nothing'):
                    #         buffer += ' '


                        # if spoken:
                            # buffer = ''
                            # spoken = False

                    candidate_votes.clear()

                # cv2.putText(
                #     frame, # image
                #     str(top_contenders), # text
                #     # for longer words, starting point is further left
                #     (100, \
                #         200), # position at which writing has to start
                #     self.FONT_FAMILY, # font family
                #     2, # font size
                #     self.FONT_COLOR, # font color
                #     self.FONT_STROKE) # font stroke

                cv2.putText(
                    frame,  # image
                    buffer,  # text
                    # for longer words, starting point is further left
                    (self.HORIZONTAL_DIST_FROM_RIGHT-(20*len(buffer)), \
                        self.VERTICAL_DIST_FROM_TOP),  # position at which writing has to start
                    self.FONT_FAMILY,  # font family
                    self.FONT_SIZE,  # font size
                    self.FONT_COLOR,  # font color
                    self.FONT_STROKE)  # font stroke

                # Step last_step: Send the image to the virtual camera for processing
                cam.send(frame)
                cam.sleep_until_next_frame()

                count += 1


if __name__ == '__main__':
    asl = ASL()
    asl.run()
