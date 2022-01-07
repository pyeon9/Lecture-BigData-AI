import os
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

videos = os.listdir('video')
print(videos)

for v in videos:
    if v == 'IMG_0033.MOV':
        name, _ = os.path.splitext(v)
        try:
            os.makedirs(f'video/{name}/')
        except:
            pass

        vidcap = cv2.VideoCapture(f'video/{name}.MOV')
        success, image = vidcap.read()

        success = True
        count = 0

        while success:
            success,image = vidcap.read()
            if count % 5 == 0:
                cv2.imwrite(f"video/{name}/{name}_{count}.png", image)
                print("saved image %d.png" % count)

            if cv2.waitKey(10) == 27:
                break
            count += 1
    else:
        pass