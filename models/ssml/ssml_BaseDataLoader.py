from typing import Dict, List
import random

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from utils import keep_keys_with_greater_than_equal_k_items

from models.base_data_loader import BaseDataLoader


class SSMLBaseDataLoader(BaseDataLoader):
    def __init__(self, perc, accessible_labels, *args, **kwargs):
        super(SSMLBaseDataLoader, self).__init__(*args, **kwargs)
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
        ## DEBUG
        print("\n\nSuccessfully in SSML\n\n")
        ## DEBUG

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
                    idxs = np.random.choice(len(instance_names), int(self.perc*len(instance_names)))
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
