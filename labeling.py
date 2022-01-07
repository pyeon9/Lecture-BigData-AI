import os
import numpy as np
import glob
import cv2
import pickle

dirs = ['IMG_0030', 'IMG_0031', 'IMG_0032', 'IMG_0033']
dir = 'IMG_0031'
path = f'video/{dir}/'
pics = os.listdir(path)


go = True
k = 0
wait = 1000
labels = {}
kdict = {'num0': 48, 'num7': 55, 'num8': 56, 'num9': 57, 'space': 32, 'a': 97, 's': 115, 'd': 100, 'f': 102, 'esc': 27,
         'back': 8, 'q':113, 'w':119, 'e':101}

while go:
    delta = 1
    frames = 1
    if frames == '0':
        break

    if k >= len(os.listdir(path)):
        break

    img = cv2.imread(path + f'{dir}_{k*5}.png')
    img = cv2.resize(img, (1280, 800))
    cv2.imshow('img', img)


    key = cv2.waitKey(wait)

    if key == kdict['esc']:
        go = False
    elif key == kdict['num0']:
        category = 0
    elif key == kdict['num7']:
        category = 7
    elif key == kdict['num8']:
        category = 8
    elif key == kdict['num9']:
        category = 9

    elif key == kdict['a']:
        wait = 500
        category = 0
    elif key == kdict['s']:
        wait = 100
        category = 0
    elif key == kdict['d']:
        wait = 50
        category = 0
    elif key == kdict['f']:
        wait = 20
        category = 0

    elif key == kdict['q']:
        delta = 3
        wait = 20
        category = 0
    elif key == kdict['w']:
        delta = 5
        wait = 20
        category = 0
    elif key == kdict['e']:
        delta = 10
        wait = 20
        category = 0

    elif key == kdict['space']:
        wait = 0
        category = 0

    elif key ==  kdict['back']:
        delta = -1
        wait = 0
        category = 0

    elif key == -1:
        category = 0
        pass
    else:
        print(key)
        category = 0


    labels[f'{k*5}'] = category
    print(k*5, '\t', labels[f'{k*5}'])

    k += delta

with open(f'video/label/{dir}.pkl', 'wb') as f:
    pickle.dump(labels, f)