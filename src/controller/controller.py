""" initialize the data and run the program"""

import argparse
from model.tfl_manager import TFLManager
from view.visualize import visualize


class Controller(object):
    # Singleton class
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
            return cls._instance

    def __init__(self):
        self.pkl, self.offset, self.frames = self.init_arguments()
        self.tfl_manager = TFLManager(self.pkl)

    def run(self):
        for i in range(len(self.frames)):
            current_frame = self.tfl_manager.on_frame(self.frames[i], self.offset + i)
            visualize(*current_frame)

    def init_arguments(self):
        args = self.get_args() # get the pls path from cmdline arguments
        frames = []
        with open(args.pls) as pls_file:
            for i, line in enumerate(pls_file):
                if i == 0:
                    pkl = line[:-1]  # get the pkl path without \n
                elif i == 1:
                    offset = int(line[:-1])  # get the offset without \n
                else:
                    frames.append(line[:-1])  # append image path to frames without \n
        return pkl, offset, frames

    @staticmethod
    def get_args():
        """return the pls path from Command line arguments"""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("-p", "--pls", required=True, help="Path to pls file", type=str)
        return parser.parse_args()
