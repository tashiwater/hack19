#!/usr/bin/env python
# -*- encoding:utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import pandas as pd
import numpy as np

def padding(img):
    tmp = img[:, :]
    height, width = img.shape[:2]
    if(height > width):
        size = height
        limit = width
    else:
        size = width
        limit = height
    start = int((size - limit) / 2)
    fin = int((size + limit) / 2)
    new_img = cv2.resize(np.full((1, 1, 3), fill_value = (255,255,255), dtype = np.uint8), (size, size))
    if(size == height):
        new_img[:, start:fin] = tmp
    else:
        new_img[start:fin, :] = tmp
    return new_img
if __name__ == "__main__":   
    datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2, 
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode='nearest')

    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = current_path + "/../../data"
    train_path = data_path + "/train_raw"
    save_train_path = data_path + "/train"
    box_list_path = data_path + "/box_list.csv"
    box_df = pd.read_csv(box_list_path, header=0)

    # 前のデータを消去
    for parts_name in box_df.name:
        parts_dir_name = save_train_path + "/" + parts_name
        img_names = os.listdir(parts_dir_name)
        if len(img_names) < 1:
            continue
        for f in img_names:
            print("remove:", f)
            os.remove(parts_dir_name + '/' + f)

    #生成
    for i, parts_name in enumerate(box_df.name):
        parts_dir_name = train_path + "/" + parts_name
        img_names = os.listdir(parts_dir_name)
        print(img_names)
        for f in img_names:
            img_array = cv2.imread(parts_dir_name + '/' + f)
            img_array = padding(img_array)
            # img_array = cv2.resize(img_array, (200,200))
            # 4次元データに変換（flow()に渡すため）
            img_array = img_array.reshape((1,) + img_array.shape)
            i = 0
            for batch in datagen.flow(img_array, batch_size=1,
                                    save_to_dir=save_train_path + "/" + parts_name, save_prefix=parts_name, save_format='png'):
                i += 1
                if i == 10:
                    break  # 停止しないと無限ループ
