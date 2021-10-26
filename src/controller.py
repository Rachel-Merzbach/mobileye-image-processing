""" initialize the data and run the program"""

from tfl_manager import TFLManager
import argparse


def init():
    args = get_args() # args = pls path
    pkl = None
    frame = []
    offset = None

    with open(args.pls) as pls_file:
        for i, j in enumerate(pls_file):
            if i == 0:
                pkl = j[:-1] # get the pkl path without \n
            elif i == 1:
                offset = int(j[:-1]) # get the offset without \n
            else:
                frame.append(j[:-1]) # append image path to on_frame without \n

    tfl_manager = TFLManager(pkl)

    return tfl_manager, frame, offset


def run():
    tfl_manager, frame, offset = init()

    for i in range(len(frame)):
        tfl_manager.on_frame(frame[i], offset + i)


def get_args():
    """return the pls path from cmd arguments"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--pls", required=True, help="Path to pls file", type=str)
    return parser.parse_args()


