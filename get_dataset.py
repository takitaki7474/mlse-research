import pandas as pd
import numpy as np
import os


class create_dateset():

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
        lables.sort()
        for label in labels:
            labelpath = os.path.join(test_img_folder, label)
            images = os.listdir(labelpath)
            for image in images:
                filepath = os.path.join(labelpath, image)
                test_dataset.append((filepath, np.int32(label)))

        return test_dataset
