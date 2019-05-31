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
import get_dataset

class train():
    def __init__(self,feature_table,train_img_folder,test_img_folder,train_images):
        self.dataset_obj = get_dataset.create_dataset(feature_table)
        self.net = cnn_mynet.MyNet_6(3)
        self.train_img_folder = train_img_folder
        self.test_img_folder = test_img_folder
        self.train_images = train_images

    def create_iter(self,batchsize):
        train_dataset = self.dataset_obj.create_train_dataset(self.train_img_folder,self.train_images)
        test_dataset = self.dataset_obj.create_test_dataset(self.test_img_folder)

        x_train, y_train = self.dataset_obj.dataset_conversion(train_dataset)
        x_val, y_val = self.dataset_obj.dataset_conversion(test_dataset)

        train_data = tuple_dataset.TupleDataset(x_train, y_train)
        val_data = tuple_dataset.TupleDataset(x_val, y_val)

        train_iter = iterators.SerialIterator(train_data, batchsize)
        valid_iter = iterators.SerialIterator(val_data, batchsize, repeat=False, shuffle=False)

        return train_iter, valid_iter

    def train_dataset(self,train_iter,valid_iter,model_name,gpu_id=-1,alpha=0.001,max_epoch=300):

        self.net = L.Classifier(self.net)
        optimizer = optimizers.Adam(alpha=alpha).setup(self.net)
        updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
        trainer = training.Trainer(updater, (max_epoch, 'epoch'), out="./result/" + model_name)

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

    def model_save(self,save_model):
        save_model_path = os.path.join("./learned_model",save_model)
        serializers.save_npz(save_model_path, self.net)
