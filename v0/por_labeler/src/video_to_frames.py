import argparse
from os.path import isdir, isfile, join
from os import listdir
import cv2
import numpy as np

def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image

def convert_videos(input_dir, output_dir, frame_buffer):
    video_files = [(join(input_dir, f), f) for f in listdir(input_dir) if isfile(join(input_dir, f))]
    for (path, filename) in video_files:
        print(path)
        filename = filename.split(".")[0]
        convert_video(path, output_dir, filename, frame_buffer)

'''
Takes in
'''
def convert_video(path, output_dir, video_name, frame_buffer):
    cap = cv2.VideoCapture(path)
    imcount = 0
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        i += 1
        if ret:
            if i % frame_buffer == 0:
                (h, w) = frame.shape[:2]
                center = (w / 2, h / 2)

                # rotate the image by 180 degrees
                M = cv2.getRotationMatrix2D(center, 270, 1.0)
                frame = cv2.warpAffine(frame, M, (w, h))
                frame = autocrop(frame)
                frame = cv2.resize(frame, (128,128))

                path = output_dir + "/" + video_name + "_" + str(imcount) + ".jpg"
                print(path)
                cv2.imwrite(path, frame)
                imcount += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Convert Video into Video Frames')
    parser.add_argument('--output_dir', '-o', required=True,
        help='Folder destination of results. Make sure this folder is empty before running the script')
    parser.add_argument('--input_dir', '-i', required = True,
        help='Input folder path. Directory should contain all video files')
    parser.add_argument('--frame_buffer', '-f', default=20, type=int,
        help='Number of frames to jump')
    args = parser.parse_args()

    if not isdir(args.output_dir):
        raise Exception("Directory does not exist: " + args.output_dir)
    if not isdir(args.input_dir):
        raise Exception("Directory does not exist: " + args.input_dir)

    # os.mkdir(join(args.output_dir, 'good'))
    # os.mkdir(join(args.output_dir, 'bad'))

    convert_videos(args.input_dir, args.output_dir, args.frame_buffer)
