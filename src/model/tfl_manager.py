import pickle
from view.visualize import visualize
from model.find_tfl_lights import find_tfl_lights
import cv2
import numpy as np
from model.manage_3d_data import Manage3dData
from model.frame_container import FrameContainer
from model.build_dataset import crop_image
from tensorflow.keras.models import load_model


class TFLManager:

    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as pkl_file:
            self.__pkl_data = pickle.load(pkl_file, encoding='latin1') # open pkl_file and load the data as binary data
        self.__pp = self.__pkl_data['principle_point']
        self.__focal = self.__pkl_data['flx']
        self.__prev_frame = None
        self.__model = load_model("../data/model.h5")

    def on_frame(self, current_frame, frame_index):
        # phase 1
        candidates, auxliary = self.__get_candidates(current_frame) # get the points that are candidates to be tfls and their colors
        assert len(candidates) == len(auxliary)
        assert len(candidates) >= 0

        # phase 2
        traffic_lights, traffic_auxliary = self.__get_tfl_coordinates(current_frame, candidates, auxliary) # get the correct tfls and their colors
        assert 0 <= len(traffic_lights) <= len(candidates)
        assert len(traffic_lights) == len(traffic_auxliary)

        # phase 3
        current_frame = FrameContainer(current_frame) # define the current frame 
        current_frame.traffic_light = np.array(traffic_lights)

        if self.__prev_frame:
            # if the frame is not the first frame - define its ego motion to be the ego motion from pkl file by its index frame
            current_frame.EM = self.__pkl_data['egomotion_' + str(frame_index - 1) + '-' + str(frame_index)]
            distance = self.__get_dists(self.__prev_frame, current_frame)

        else:
            distance = None

        self.__prev_frame = current_frame
        return current_frame, candidates, auxliary, traffic_lights, traffic_auxliary, distance
        

    @staticmethod
    def __get_candidates(image):
        # get the path of image

        x_red, y_red, x_green, y_green = find_tfl_lights(cv2.imread(image))
        assert len(x_red) == len(y_red)
        assert len(x_green) == len(y_green)

        candidates = [[x_red[i], y_red[i]] for i in range(len(x_red))] + [[x_green[i], y_green[i]] for i in range(len(x_green))]
        auxliary = ["red" for _ in x_red] + ["green" for _ in x_green]
        
        # candidates[i] is coordinate (x,y) and auxliary[i] represent it's color
        return candidates, auxliary


    def __get_tfl_coordinates(self, image, candidates, auxliary):

        crop_shape = (81, 81) # cropped according to the requirements
        l_predicted_label = [] # l_predicted_label[i] = 1 if the model decide that candidates[i] is tfl else 0

        for candidate in candidates:
            crop_img = crop_image(cv2.imread(image), candidate[0], candidate[1]) # crop the image
            predictions = self.__model.predict(crop_img.reshape([-1] + list(crop_shape) + [3])) # send the image to model
            l_predicted_label.append(1 if predictions[0][1] > 0.98 else 0) # determine 1 or 0 for the candidate

        traffic_lights = [candidates[i] for i in range(len(candidates)) if l_predicted_label[i] == 1]
        auxliary = [auxliary[i] for i in range(len(auxliary)) if l_predicted_label[i] == 1]

        return traffic_lights, auxliary


    def __get_dists(self, prev_frame, current_frame):
        manage3dData = Manage3dData(prev_frame, current_frame, self.__focal, self.__pp)
        current_frame = manage3dData.calc_TFL_dist()

        return np.array(current_frame.tfls_3d_location)[:, 2] # return array of dists

