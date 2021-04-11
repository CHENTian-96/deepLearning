# -*- coding: utf-8 -*- 
# @Time : 2021/4/3 21:55 
# @Author : CHENTian
# @File : get_img_name.py

import os

def get_image_path(image_path, storage_path):
    with open(storage_path, "w") as f:
        index = 0
        for cur_dir, dirs, files in os.walk(image_path):
            if not dirs and files:
                for name in files:
                    f.write(cur_dir.strip(".") + "/" + name + " " + str(index) + "\n")
                    print(cur_dir.strip(".") + "/" + name + " " + str(index))
                index += 1

train_path = "coco_animals/train"
train_txt = "./train.txt"
test_path = "coco_animals/val"
test_txt = "./test.txt"

get_image_path(train_path, train_txt)
get_image_path(test_path, test_txt)