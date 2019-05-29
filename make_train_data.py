import numpy as np
import pandas as pd
import cv2
import os

# npyファイルの配列から訓練リストを生成
def make_train_list(input_img_folder, pkl_path, np_file_path):

    train_list = []
    image_list = []
    img_list = np.load(np_file_path)
    df = pd.read_pickle(pkl_path)

    for img_name in img_list:
        label = df[df.img == img_name].label.values[0]
        filepath = input_img_folder + "/" + str(label) + "/" + img_name
        train_list.append((filepath, np.int32(label)))
        image_list.append(img_name)

    print("画像枚数：{}".format(len(train_list)))

    return train_list, image_list

def make_val_list(input_img_folder, pkl_path, np_file_path):

    val_list = []
    image_list = []
    val_img_list = []
    img_list = np.load(np_file_path)
    df = pd.read_pickle(pkl_path)
    remove_index = []
    val_list_num = len(img_list) * 0.2

    for img_name in img_list:
        remove_index.append(df[df.img == img_name].index[0])

    removed_df = df.drop(index=remove_index)

    df_0 = removed_df[removed_df.label == 0]
    df_1 = removed_df[removed_df.label == 1]
    df_2 = removed_df[removed_df.label == 2]

    sample_df_0 = df_0.sample(n=int(val_list_num/3))
    sample_df_1 = df_1.sample(n=int(val_list_num/3))
    sample_df_2 = df_2.sample(n=int(val_list_num/3))

    val_img_list.extend(sample_df_0.img.values)
    val_img_list.extend(sample_df_1.img.values)
    val_img_list.extend(sample_df_2.img.values)


    for img_name in val_img_list:
        label = df[df.img == img_name].label.values[0]
        filepath = input_img_folder + "/" + str(label) + "/" + img_name
        val_list.append((filepath, np.int32(label)))
        image_list.append(img_name)

    print("評価データ数：{}".format(len(val_list)))

    return val_list, image_list

def make_test_data(test_folder_path):

    test_list = []
    test_dir = os.listdir(test_folder_path)
    test_dir.sort()
    for label in test_dir:
        labelpath = os.path.join(test_folder_path, label)
        images = os.listdir(labelpath)
        for image in images:
            imagepath = os.path.join(labelpath, image)
            test_list.append((imagepath, np.int32(label)))

    print("評価データ数：{}".format(len(test_list)))

    return test_list


def make_dataset(data_list):

    x_data = []
    y_data = []

    for filepath, label in data_list:
        img = cv2.imread(filepath, 1)
        img = cv2.resize(img,(28,28))
        x_data.append(img)
        y_data.append(label)

    x_data = np.array(x_data).astype(np.float32)
    y_data = np.array(y_data).astype(np.int32)

    return x_data, y_data
