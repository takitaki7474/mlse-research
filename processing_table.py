import pandas as pd
import numpy as np

class processing_feature_table():

    def __init__(self,feature_table):
        self.feature_table = feature_table
        self.class_num = len(set(self.feature_table["label"]))

    def table_division(self):
        devided_df = []
        for label, df in self.feature_table.groupby("label"):
            devided_df.append(df)

        return devided_df

    def add_feature_sum(self):
        feature_sum_values = []
        for i in self.feature_table["feature"]:
            feature_sum_values.append(sum(i))
        self.feature_table["feature_sum"] = feature_sum_values

    def add_feature_highorder_sum(self, highorder):
        column_name = "highorder_sum_" + str(highorder)
        highorder_sum = []
        for i in self.feature["feature"]:
            sorted_feature = sorted(i, reverse=True)
            highorder_sum.append(sum(sorted_feature[:highorder]))
        self.feature[column_name] = highorder_sum

    def feature_table_sort(self,sort_column):
        self.feature = self.feature.sort_values(by=[sort_column], ascending=True)
        self.feature = self.feature.reset_index()

    def search_img(self,img_num):
        images = list(self.feature["img"][:img_num])

        return images
