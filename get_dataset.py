import pandas as pd
import numpy as np
import os
import cv2


class create_dataset():

    def __init__(self,feature_table):
        self.feature_table = feature_table

    def create_train_dataset(self,train_img_folder,images):
        train_dataset = []
        for image in images:
            label = self.feature_table[self.feature_table.img == image].label.values[0]
            filepath = os.path.join(train_img_folder,str(label),image)
            train_dataset.append((filepath, np.int32(label)))

        return train_dataset

    def create_test_dataset(self,test_img_folder):
        test_dataset = []
        labels = os.listdir(test_img_folder)
        labels.sort()
        for label in labels:
            labelpath = os.path.join(test_img_folder, label)
            images = os.listdir(labelpath)
            for image in images:
                filepath = os.path.join(labelpath, image)
                test_dataset.append((filepath, np.int32(label)))

        return test_dataset

    def dataset_conversion(self,dataset):
        img_data = []
        label_data = []

        for filepath, label in dataset:
            img = cv2.imread(filepath, 1)
            img = cv2.resize(img,(28,28))
            img_data.append(img)
            label_data.append(label)

        img_data = np.array(img_data).astype(np.float32)
        label_data = np.array(label_data).astype(np.int32)

        return img_data, label_data
