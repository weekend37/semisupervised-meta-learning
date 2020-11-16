from abc import abstractmethod
import numpy as np
import os
from tqdm import tqdm

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List

import sys
from path import mypath
sys.path.append(mypath)

from utils import combine_first_two_axes, keep_keys_with_greater_than_equal_k_items
import settings

from models.ssml.ssml_BaseDataLoader import SSMLBaseDataLoader
from models.lasiummamlgan.maml_gan import MAMLGAN
from models.maml.maml import ModelAgnosticMetaLearningModel

class SSMLMAML(ModelAgnosticMetaLearningModel):
    def __init__(self, perc, accessible_labels, *args, **kwargs):
        super(SSMLMAML, self).__init__(*args, **kwargs)
        self.perc = perc
        self.accessible_labels = accessible_labels
        data_loader_cls = SSMLBaseDataLoader
        self.data_loader = self.init_ssml_data_loader(data_loader_cls)

    def init_ssml_data_loader(self, data_loader_cls):
        return data_loader_cls(
            perc=self.perc,
            accessible_labels=self.accessible_labels,
            database=self.database,
            val_database=self.val_database,
            test_database=self.test_database,
            n=self.n,
            k_ml=self.k_ml,
            k_val_ml=self.k_val_ml,
            k_val=self.k_val,
            k_val_val=self.k_val_val,
            k_test=self.k_test,
            k_val_test=self.k_val_test,
            meta_batch_size=self.meta_batch_size,
            num_tasks_val=self.num_tasks_val,
            val_seed=self.val_seed
        )


class SSMLMAMLGAN(MAMLGAN):
    def __init__(self, perc, accessible_labels, ssml_maml, *args, **kwargs):
        super(SSMLMAMLGAN, self).__init__(*args, **kwargs)
        self.perc = perc
        self.accessible_labels = accessible_labels
        self.ssml_maml = ssml_maml

    def train(self, iterations=5):
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)

        maml_train_dataset = self.ssml_maml.get_train_dataset()
        maml_gan_train_dataset = self.get_train_dataset()

        iteration_count = self.load_model()

        N = 20
        N_labeled = int(self.perc * N)
        N_gen = int(N - N_labeled)
        
        print(" #### ### ## # N_labeled:", N_labeled)
        print(" #### ### ## # N_gen", N_gen)
        print("Percentage of labeled data:", self.perc)

        epoch_count = iteration_count // N

        pbar = tqdm(maml_train_dataset)

        train_accuracy_metric = tf.metrics.Mean()
        train_accuracy_metric.reset_states()
        train_loss_metric = tf.metrics.Mean()
        train_loss_metric.reset_states()

        should_continue = iteration_count < iterations
        while should_continue:

            for d, dataset in enumerate([maml_gan_train_dataset, maml_train_dataset]):

                # print("DATASET:", ["generated data", "labeled data"][d])
            
                N_dataset = [N_gen, N_labeled][d]
                iteration_count_inner = 0
                should_continue_inner = iteration_count_inner < N_dataset
                
                while should_continue_inner:
                    for (train_ds, val_ds), (train_labels, val_labels) in dataset:
                        
                        train_acc, train_loss = self.meta_train_loop(train_ds, val_ds, train_labels, val_labels)
                        train_accuracy_metric.update_state(train_acc)
                        train_loss_metric.update_state(train_loss)
                        iteration_count += 1
                        if (
                            self.log_train_images_after_iteration != -1 and
                            iteration_count % self.log_train_images_after_iteration == 0
                        ):
                            self.log_images(
                                self.train_summary_writer,
                                combine_first_two_axes(train_ds[0, ...]),
                                combine_first_two_axes(val_ds[0, ...]),
                                step=iteration_count
                            )
                            self.log_histograms(step=iteration_count)

                        if iteration_count != 0 and iteration_count % self.save_after_iterations == 0:
                            self.save_model(iteration_count)

                        if iteration_count % self.report_validation_frequency == 0:
                            self.report_validation_loss_and_accuracy(iteration_count)
                            if iteration_count != 0:
                                print('Train Loss: {}'.format(train_loss_metric.result().numpy()))
                                print('Train Accuracy: {}'.format(train_accuracy_metric.result().numpy()))
                            with self.train_summary_writer.as_default():
                                tf.summary.scalar('Loss', train_loss_metric.result(), step=iteration_count)
                                tf.summary.scalar('Accuracy', train_accuracy_metric.result(), step=iteration_count)
                            train_accuracy_metric.reset_states()
                            train_loss_metric.reset_states()

                        pbar.set_description_str('Epoch{}, Iteration{}: Train Loss: {}, Train Accuracy: {}'.format(
                            epoch_count,
                            iteration_count,
                            train_loss_metric.result().numpy(),
                            train_accuracy_metric.result().numpy()
                        ))
                        pbar.update(1)

                        iteration_count_inner += 1
                        # which_data = ['gen_data','label_data'][d]
                        # print(f"\n\nThis is the {iteration_count_inner}th iteration in {which_data}\n")
                        if iteration_count_inner >= N_dataset:
                            should_continue_inner = False
                            break

            if iteration_count >= iterations:
                should_continue = False
                break

            epoch_count += 1
