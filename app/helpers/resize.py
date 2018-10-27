#!/usr/bin/python
from PIL import Image
import os
import sys

path = "/Users/oliver/Developer/FinalApp/scripts/tf_files/cells/8micron_bead/"
dirs = os.listdir(path)


def resize():
    for item in dirs:
        if os.path.isfile(path + item):
            full = os.path.join(path + item)
            print(full)
            if full.endswith('.jpg'):
                print('hi')

                im = Image.open(path + item)
                f, e = os.path.splitext(path + item)
                imResize = im.resize((128, 128), Image.ANTIALIAS)
                imResize.save(f + ' 128.jpg', 'JPEG', quality=100)


def changeFormat():
    for item in dirs:
        if os.path.isfile(path + item):
            print('hi')
            full = os.path.join(path + item)
            if full.endswith('.png'):
                im = Image.open(full)
                full = full[:-4]
                full = full + '.jpg'
                im.save(full)
"""


resize()
