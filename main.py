#-*- coding:utf-8 -*-
import train
import processing_table
import os
import pandas as pd
import argparse


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, help='model name')
    parser.add_argument('--pkl_name', '-pkl', type=str, default='feature_v5_fc7.pkl', help='feature table name')
    parser.add_argument('--train_img_folder_path', '-train', default='./train_img', type=str, help='Train image folder path')
    parser.add_argument('--test_img_folder_path', '-test', default='./test_img', type=str, help='Test image folder path')
    parser.add_argument('--batchsize', '-b', default=128, type=int, help='batchsize to train')
    parser.add_argument('--gpu_id', '-g', default=-1, type=int, help='use gpu or cpu')
    parser.add_argument('--epoch', '-e', type=int, default=300, help='Number of epoch to train')
    parser.add_argument('--alpha', '-a', type=float, default=0.001, help='Number of alpha to train')
    args = parser.parse_args()

    model_name = args.model_name
    feature_table_path = os.path.join("./feature",args.pkl_name)
    feature_table = pd.read_pickle(feature_table_path)
    train_img_folder_path = args.train_img_folder_path
    test_img_folder_path = args.test_img_folder_path
    batchsize = args.batchsize
    gpu_id = args.gpu_id
    epoch = args.epoch
    alpha = args.alpha

    return model_name, feature_table, train_img_folder_path, test_img_folder_path, batchsize, gpu_id, epoch, alpha


if __name__=="__main__":
    model_name, feature_table, train_img_folder_path, test_img_folder_path, batchsize, gpu_id, epoch, alpha = arg_parse()

    save_model_name = model_name
    highorder = 100
    img_num = 100
    image_set = []

    table_obj = processing_table.processing_feature_table(feature_table)
    column_name = table_obj.add_feature_highorder_sum(highorder)
    devided_df = table_obj.table_division()
    for df in devided_df:
        df = table_obj.feature_table_sort(df,column_name,ascending=False)
        image_set += table_obj.over_search_img(df,img_num)

    print("訓練データ数: {}".format(len(image_set)))

    train_obj = train.train(feature_table, train_img_folder_path, test_img_folder_path, image_set)
    train_iter, valid_iter = train_obj.create_iter(batchsize)
    train_obj.train_dataset(train_iter, valid_iter, model_name, gpu_id=gpu_id, alpha=alpha, max_epoch=epoch)
    train_obj.model_save(save_model_name)
