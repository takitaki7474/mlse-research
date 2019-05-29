# -*- coding:utf-8 -*-

import numpy as np
import cv2
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import iterators, training, optimizers, serializers
from chainer.datasets import tuple_dataset, split_dataset_random
from chainer.training import extensions
import argparse
import os
import cnn_mynet
import make_train_data

class train():

    def __init__(self):

        parser = argparse.ArgumentParser(description='Train Sample')
        parser.add_argument('--train_list', '-train', default='./cut_img', type=str, help='Train image folder name')
        parser.add_argument('--model_name', '-m', default='v1.model', type=str, help='model name')
        parser.add_argument('--epoch', '-e', type=int, default=80, help='Number of epochs to train')
        parser.add_argument('--batchsize', '-b', type=int, default=128, help='Number of batchsize to train')
        parser.add_argument('--alpha', '-a', type=float, default=0.001, help='Number of alpha to train')
        parser.add_argument('--numpy_file', '-np', type=str, default='random.npy', help='Number of data to train')
        parser.add_argument('--pkl_file', '-pkl', type=str, default='feature_v2.pkl', help='Number of data to train')
        args = parser.parse_args()

        self.img_folder = args.train_list
        self.max_epoch = args.epoch
        self.batchsize = args.batchsize
        self.alpha = args.alpha
        self.model_name = args.model_name
        self.numpy_file = args.numpy_file
        self.pkl_file = args.pkl_file

        self.np_file_path = os.path.join("./npy_files", self.numpy_file)
        self.pkl_file_path = os.path.join("./feature", self.pkl_file)
        self.save_model_path = os.path.join("./learned_model", self.model_name)
        self.save_model_path = self.save_model_path + ".model"
        self.test_folder_path = "./test_img_v2"

    def create_dataset(self):

        train_list, train_image_list = make_train_data.make_train_list(self.img_folder, self.pkl_file_path, self.np_file_path)
        val_list = make_train_data.make_test_data(self.test_folder_path)

        x_train, y_train = make_train_data.make_dataset(train_list)
        x_val, y_val = make_train_data.make_dataset(val_list)

        train_data = tuple_dataset.TupleDataset(x_train, y_train)
        val_data = tuple_dataset.TupleDataset(x_val, y_val)

        train_iter = iterators.SerialIterator(train_data, batchsize)
        valid_iter = iterators.SerialIterator(val_data, batchsize, repeat=False, shuffle=False)

        return train_iter, valid_iter

    def train_dataset(self,train_iter,valid_iter):

        gpu_id = -1
        self.net = cnn_mynet.MyNet_6(3)
        self.net = L.Classifier(self.net)
        optimizer = optimizers.Adam(alpha=self.alpha).setup(self.net)
        updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
        trainer = training.Trainer(updater, (self.max_epoch, 'epoch'), out="./result/" + self.model_name)

        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
        trainer.extend(extensions.Evaluator(valid_iter, self.net, device=gpu_id), name="val")
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
        trainer.extend(extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std.png'))
        trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
        #trainer.extend(extensions.PlotReport(['val/main/loss'], x_key='epoch', file_name='loss.png'))
        #trainer.extend(extensions.PlotReport(['val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
        trainer.extend(extensions.dump_graph('main/loss'))

        trainer.run()

    def model_save(self):
        serializers.save_npz(self.save_model_path, self.net)

if __name__=="__main__":
    train_obj = train()
    train_iter, valid_iter = train_obj.create_dataset()
    train_obj.train_dataset(train_iter,valid_iter)
    train_obj.model_save()
