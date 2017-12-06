'''
Simple opencv userinterface that will label images as "good" or "bad"
'''

import argparse
from os.path import isdir, isfile, join
from os import listdir
import cv2

'''
NEW IDEA:
Convert video to frames, and then store them all in our directory.
This also allows for more flexibility
'''
def read_frames(path, output_dir):
    frame_files = [(join(path, f),f) for f in listdir(path) if isfile(join(path, f))]
    total = len(frame_files)
    count = 0
    for (path, filename) in frame_files:
        frame = cv2.imread(path)
        cv2.imshow('frame', frame)
        quit = False
        while True:
            ch = cv2.waitKey()
            if ch & 0xFF == ord('q'):
                quit = True
                break
            elif ch & 0xFF == ord('0'):
                path = output_dir + "/bad/" + filename
                cv2.imwrite(path, frame)
                break
            elif ch & 0xFF == ord('1'):
                path = output_dir + "/neutral/" + filename
                cv2.imwrite(path, frame)
                break
            elif ch & 0xFF == ord('2'):
                path = output_dir + "/good/" + filename
                cv2.imwrite(path, frame)
                break
        count += 1
        print("Progress:", count / total * 100)
        if quit:
            break

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Annotate Video Frames')
    parser.add_argument('--output_dir', '-o', required=True,
        help='Folder destination of results. Make sure this folder is empty before running the script')
    parser.add_argument('--input_dir', '-i', required = True,
        help='Input folder path. Directory should contain all video frame files')
    args = parser.parse_args()

    if not isdir(args.output_dir):
        raise Exception("Directory does not exist: " + args.output_dir)
    if not isdir(args.input_dir):
        raise Exception("Directory does not exist: " + args.input_dir)

    read_frames(args.input_dir, args.output_dir)
