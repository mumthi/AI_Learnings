import os

import numpy as np
import random
from matplotlib import pyplot as plt

number_of_images = 1000
width = 9
height = 9
path = r'D:\learning\AI_Learnings\tries\Resorces\synthetic_image'
for i in range(number_of_images):
    im = np.zeros((height, width))
    im_path = os.path.join(path, str(i))
    for j in range(width):
        for k in range(height):
            im[j, k] = random.choice((0, 1))
    plt.imshow(im)
    plt.savefig(im_path)
