import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import time
import sys
from path import mypath
sys.path.append(mypath)

from databases import OmniglotDatabase
from models.lasiummamlgan.database_parsers import OmniglotParser

from models.lasiummamlgan.gan import GAN
from models.lasiummamlgan.maml_gan import MAMLGAN
from models.maml.maml import ModelAgnosticMetaLearningModel
from networks.maml_umtra_networks import SimpleModel, MiniImagenetModel, VGG19Model, FiveLayerResNet

from models.lasiummamlgan.maml_gan_omniglot import get_generator, get_discriminator
from models.ssml.sl_maml_gan import SSMLMAML, SSMLMAMLGAN


if __name__ == '__main__':

    if len(sys.argv) == 1:
        print("Usage:")
        print("   $ python3 ssml_maml_gan_omniglot.py <labeled_percentage>")
        print("where <labeled_percentage> is a number in [0,1] (preferable having 0 or 5 in 2nd decimal place like 0, 0.05, 0.10, etc..)")
        sys.exit(9)
    elif len(sys.argv) == 2:
        labeled_percentage = float(sys.argv[1])
        prefix = 'sl_new_scheme_omniglot_perc'
    elif len(sys.argv) == 3:
        labeled_percentage = float(sys.argv[1])
        prefix = sys.argv[2]
    else:
        print("Error: Too many arguments")
        sys.exit(0)

    # CONFIGS
    ITERATIONS = 1000
    GAN_EPOCHS = 500
    N_TASK_EVAL = 100
    K = 5
    N_WAY = 12
    META_BATCH_SIZE = 1
    LASIUM_TYPE = None

    print("K=",K)
    print("N_WAY=",N_WAY)

    omniglot_database = OmniglotDatabase(random_seed=47, num_train_classes=1200, num_val_classes=100)
    shape = (28, 28, 1)
    latent_dim = 128
    omniglot_generator = get_generator(latent_dim)
    omniglot_discriminator = get_discriminator()
    omniglot_parser = OmniglotParser(shape=shape)

    experiment_name = prefix+str(labeled_percentage)

    L = None

    gan = GAN(
        'omniglot',
        image_shape=shape,
        latent_dim=latent_dim,
        database=omniglot_database,
        parser=omniglot_parser,
        generator=omniglot_generator,
        discriminator=omniglot_discriminator,
        visualization_freq=50,
        d_learning_rate=0.0003,
        g_learning_rate=0.0003,
    )
    gan.perform_training(epochs=GAN_EPOCHS, checkpoint_freq=50)
    gan.load_latest_checkpoint()

    print("GAN training finished")
    time.sleep(1)

    train_folders = omniglot_database.train_folders
    print("before:", len(list(train_folders.keys())))
    keys = list(train_folders.keys())
    sample_size = np.max([N_WAY*META_BATCH_SIZE, int(len(train_folders.keys())*labeled_percentage)])
    keys = np.random.choice(keys, sample_size, replace=False)
    train_folders = {k: v for (k, v) in train_folders.items() if k in keys}
    print("after:", len(list(train_folders.keys())))
    omniglot_database.train_folders = train_folders
    
    ssml_maml = SSMLMAML(

        perc=labeled_percentage,
        accessible_labels=L,

        database=omniglot_database,
        network_cls=SimpleModel,
        n=N_WAY,
        k_ml=K,
        k_val_ml=K,
        k_val=K,
        k_val_val=K,
        k_test=K,
        k_val_test=K,
        meta_batch_size=META_BATCH_SIZE,
        num_steps_ml=5,
        lr_inner_ml=0.4,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=1001,
        num_tasks_val=100,
        clip_gradients=False,
        experiment_name='omniglot',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    ssml_maml_gan = SSMLMAMLGAN(

        perc=labeled_percentage,
        accessible_labels=L,
        ssml_maml=ssml_maml,

        gan=gan,
        latent_dim=latent_dim,
        generated_image_shape=shape,

        lasium_type=LASIUM_TYPE,

        database=omniglot_database,
        network_cls=SimpleModel,
        n=N_WAY,
        k_ml=K,
        k_val_ml=K,
        k_val=K,
        k_val_val=K,
        k_val_test=K,
        k_test=K,
        meta_batch_size=META_BATCH_SIZE,
        num_steps_ml=5,
        lr_inner_ml=0.4,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=1001,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        clip_gradients=False,
        experiment_name=experiment_name,
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    ssml_maml_gan.train(iterations=ITERATIONS)
    ssml_maml_gan.evaluate(50, N_TASK_EVAL, seed=94305)
