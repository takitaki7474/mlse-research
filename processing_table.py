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
