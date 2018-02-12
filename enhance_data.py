from shutil import copy2
from face_helper import list_files
import cv2

src_dir = "G:/FakeAppData/128/taohong128/"
dst_dir = "G:/FakeAppData/128_enhance/taohong128/"
import numpy as np


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

import traceback
import sys

def enhance_image_set(src_dir, dst_dir):
    files = list_files(src_dir)
    for file in files:
        try:
            # copy first
            filename = file.split("/")[-1]
            filename = filename.split(".")[0]

            copy2(file, dst_dir)
            img = cv2.imread(file)
            flipped = np.fliplr(img)
            rotated = rotateImage(img, 10)
            rotated2 = rotateImage(img, 350)

            cv2.imwrite(dst_dir + filename + "_flip.png", img=flipped)
            cv2.imwrite(dst_dir + filename + "_rotate1.png", img=rotated)
            cv2.imwrite(dst_dir + filename + "_rotate2.png", img=rotated2)
            print("Done : "+ filename)
        except Exception as ex:

            print(traceback.format_exc())
            # or
            print(sys.exc_info()[0])
            print("cannot process file : " + file)

enhance_image_set(src_dir,dst_dir)