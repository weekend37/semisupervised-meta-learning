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

from models.lasiummamlgan.maml_gan import MAMLGAN
from models.maml.maml import ModelAgnosticMetaLearningModel

class SSMLMAML(ModelAgnosticMetaLearningModel):
    def __init__(self, perc, accessible_labels, *args, **kwargs):
        super(SSMLMAML, self).__init__(*args, **kwargs)
        self.perc = perc
        self.accessible_labels = accessible_labels

    def get_supervised_meta_learning_dataset(
        self,
        folders: Dict[str, List[str]],
        n: int,
        k: int,
        k_validation: int,
        meta_batch_size: int,
        one_hot_labels: bool = True,
        reshuffle_each_iteration: bool = True,
        seed: int = 94305,
        dtype=tf.float32,  # The input dtype
        instance_parse_function=None,
        buffer_size=1
    ) -> tf.data.Dataset:
        """
            Folders are dictionary
            If it is a dictionary then each item is the class name and the corresponding values are the file addresses
            of images of that class.
        """
        if instance_parse_function is None:
            instance_parse_function = self.get_parse_function()

        # if seed != -1:
        #     np.random.seed(seed)

        def _get_instances(class_dir_address):
            def get_instances(class_dir_address):
                class_dir_address = class_dir_address.numpy().decode('utf-8')
                instance_names = folders[class_dir_address]
                print(instance_names[0])
                print(instance_names)

                if self.accessible_labels is None:
                    # make sure we only have a limited subset of data available
                    np.random.seed(seed)
                    idxs = np.random.choice(len(instance_names),int(self.perc*len(instance_names)))
                    instances = np.random.choice(instance_names[idxs], size=k + k_validation, replace=False)
                else:
                    print("TODO: IMPLEMENT USAGE OF ACCESSIBLE LABELS")
                    # TODO: Implement usage of accessible labels
                    # maybe somethine like:
                    # instances = self.accessible_labels[class_dir_address]

                return instances[:k], instances[k:k + k_validation]

            return tf.py_function(get_instances, inp=[class_dir_address], Tout=[tf.string, tf.string])

        if seed != -1:
            parallel_iterations = 1
        else:
            parallel_iterations = None

        def parse_function(tr_imgs_addresses, val_imgs_addresses):
            tr_imgs = tf.map_fn(
                instance_parse_function,
                tr_imgs_addresses,
                dtype=dtype,
                parallel_iterations=parallel_iterations
            )
            val_imgs = tf.map_fn(
                instance_parse_function,
                val_imgs_addresses,
                dtype=dtype,
                parallel_iterations=parallel_iterations
            )

            return tf.stack(tr_imgs), tf.stack(val_imgs)

        keep_keys_with_greater_than_equal_k_items(folders, k + k_validation)

        dataset = tf.data.Dataset.from_tensor_slices(sorted(list(folders.keys())))
        if seed != -1:
            dataset = dataset.shuffle(
                # buffer_size=buffer_size,
                buffer_size=len(folders.keys()),
                reshuffle_each_iteration=reshuffle_each_iteration,
                seed=seed
            )
            # When using a seed the map should be done in the same order so no parallel execution
            dataset = dataset.map(_get_instances, num_parallel_calls=1)
        else:
            dataset = dataset.shuffle(
                # buffer_size=buffer_size,
                buffer_size=len(folders.keys()),
                reshuffle_each_iteration=reshuffle_each_iteration
            )
            dataset = dataset.map(_get_instances, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(n, drop_remainder=True)

        labels_dataset = self.make_labels_dataset(n, k, k_validation, one_hot_labels=one_hot_labels)

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))

        steps_per_epoch = tf.data.experimental.cardinality(dataset)
        if steps_per_epoch == 0:
            dataset = dataset.repeat(-1).take(meta_batch_size).batch(meta_batch_size)
        else:
            dataset = dataset.batch(meta_batch_size, drop_remainder=True)

        return dataset

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

        iteration_count = self.load_model() # this thing might be questionable (?)

        # N_labeled = tf.data.experimental.cardinality(maml_train_dataset)

        N_labeled = len(maml_train_dataset)
        print(" #### ### ## # N_labeled:", N_labeled)
        N = N_labeled // self.perc # effective dataset length 3/0.5 = 6
        N_gen = int(N - N_labeled) # 6-3
        epoch_count = iteration_count // N

        pbar = tqdm(maml_train_dataset)

        train_accuracy_metric = tf.metrics.Mean()
        train_accuracy_metric.reset_states()
        train_loss_metric = tf.metrics.Mean()
        train_loss_metric.reset_states()

        should_continue = iteration_count < iterations
        while should_continue:
            
            DS = [maml_gan_train_dataset, maml_train_dataset]
            for d, dataset in enumerate(DS):

                print("DATASET:", ["generated data", "labeled data"][d])
            
                N_dataset = [N_gen, N_labeled][d] 
                for _ in range(N_dataset):

                    # take didn't work, this just queries the generator for one meta detaset
                    (train_ds, val_ds), (train_labels, val_labels) = iter(dataset).next() 
                    
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
                

            if iteration_count >= iterations:
                should_continue = False
                break

            epoch_count += 1
