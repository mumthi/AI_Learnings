import os

import numpy as np
import random

from cv2 import cv2

number_of_images = 100
width = 9
height = 9
path = [r'D:\learning\AI_Learnings\tries\images\good',
        r'D:\learning\AI_Learnings\tries\images\bad']
for i in range(number_of_images):
    im = np.zeros((height, width))
    if i < int(number_of_images/2):
        im_path = os.path.join(path[0], str(i)+".png")
    else:
        im_path = os.path.join(path[1], str(i)+".png")
    for j in range(width):
        for k in range(height):
            im[j, k] = random.choice((0, 255))
    cv2.imwrite(im_path, im)
