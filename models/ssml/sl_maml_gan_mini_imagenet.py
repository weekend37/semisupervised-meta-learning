import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import time
import sys
from path import mypath
sys.path.append(mypath)

from databases import MiniImagenetDatabase
from models.lasiummamlgan.database_parsers import MiniImagenetParser

from models.lasiummamlgan.gan import GAN
from models.lasiummamlgan.maml_gan import MAMLGAN
from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import SimpleModel, MiniImagenetModel, VGG19Model, FiveLayerResNet

from models.lasiummamlgan.maml_gan_mini_imagenet import get_generator, get_discriminator
from models.ssml.sl_maml_gan import SSMLMAML, SSMLMAMLGAN

if __name__ == '__main__':
    
    if len(sys.argv) == 1:
        print("Usage:")
        print("   $ python3 sl_maml_gan_mini_imagenet.py <labeled_percentage>")
        print("where <labeled_percentage> is a number in [0,1] (preferable having 0 or 5 in 2nd decimal place like 0, 0.05, 0.10, etc..)")
        sys.exit(9)
    elif len(sys.argv) == 2:
        labeled_percentage = float(sys.argv[1])
        prefix = 'sl_new_scheme_mini_imagenet_perc'
    elif len(sys.argv) == 3:
        labeled_percentage = float(sys.argv[1])
        prefix = sys.argv[2]
    else:
        print("Error: Too many arguments")
        sys.exit(0)

    # CONFIGS
    ITERATIONS = 15000
    GAN_EPOCHS = 100
    N_TASK_EVAL = 100
    K = 5
    N_WAY = 5
    META_BATCH_SIZE = 1
    LASIUM_TYPE = "p1"

    print("K=",K)
    print("N_WAY=",N_WAY)

    mini_imagenet_database = MiniImagenetDatabase()
    shape = (84, 84, 3)
    latent_dim = 512
    mini_imagenet_generator = get_generator(latent_dim)
    mini_imagenet_discriminator = get_discriminator()
    mini_imagenet_parser = MiniImagenetParser(shape=shape)

    experiment_name = prefix+str(labeled_percentage)

    # for the SSGAN we need to feed the labels, L, when initializing
    gan = GAN(
        'mini_imagenet',
        image_shape=shape,
        latent_dim=latent_dim,
        database=mini_imagenet_database,
        parser=mini_imagenet_parser,
        generator=mini_imagenet_generator,
        discriminator=mini_imagenet_discriminator,
        visualization_freq=1,
        d_learning_rate=0.0003,
        g_learning_rate=0.0003,
    )
    gan.perform_training(epochs=GAN_EPOCHS, checkpoint_freq=50)
    gan.load_latest_checkpoint()

    print("training GAN is done")
    time.sleep(1)

    # Split labeled and not labeled
    train_folders = mini_imagenet_database.train_folders
    keys = list(train_folders.keys())
    labeled_keys = np.random.choice(keys, int(len(train_folders.keys())*labeled_percentage), replace=False)
    train_folders_labeled = {k: v for (k, v) in train_folders.items() if k in labeled_keys}
    train_folders_unlabeled = {k: v for (k, v) in train_folders.items() if k not in labeled_keys}
    mini_imagenet_database.train_folders = train_folders_labeled

    L = None
    ssml_maml = SSMLMAML(
        
        perc=labeled_percentage,
        accessible_labels = L, 

        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=N_WAY,
        k_ml=K,
        k_val_ml=K,
        k_val=K,
        k_val_val=K,
        k_test=K,
        k_val_test=K,
        meta_batch_size=META_BATCH_SIZE,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=15000,
        meta_learning_rate=0.001,
        report_validation_frequency=1000,
        log_train_images_after_iteration=1000,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='mini_imagenet',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    ssml_maml_gan = SSMLMAMLGAN(

        perc=labeled_percentage,
        accessible_labels = L, 
        ssml_maml=ssml_maml,

        gan=gan,
        latent_dim=latent_dim,
        generated_image_shape=shape,
        
        lasium_type = LASIUM_TYPE,

        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=N_WAY,
        k_ml=K,
        k_val=K,
        k_val_ml=K,
        k_val_val=K,
        k_val_test=K,
        k_test=K,
        meta_batch_size=META_BATCH_SIZE,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        clip_gradients=False,
        experiment_name='mini_imagenet',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    ssml_maml_gan.train(iterations=ITERATIONS)
    ssml_maml_gan.evaluate(50, N_TASK_EVAL, seed=94305)

