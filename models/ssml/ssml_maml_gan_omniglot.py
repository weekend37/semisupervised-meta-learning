import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers

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
from models.ssml.ssml_maml_gan import SSMLMAML, SSMLMAMLGAN


if __name__ == '__main__':
    omniglot_database = OmniglotDatabase(random_seed=47, num_train_classes=1200, num_val_classes=100)
    shape = (28, 28, 1)
    latent_dim = 128
    omniglot_generator = get_generator(latent_dim)
    omniglot_discriminator = get_discriminator()
    omniglot_parser = OmniglotParser(shape=shape)
    labeled_percentage = 0.5
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
    gan.perform_training(epochs=1, checkpoint_freq=50)
    gan.load_latest_checkpoint()

    print("training GAN is done")
    time.sleep(1)

    ssml_maml = SSMLMAML(

        perc=labeled_percentage,
        accessible_labels=L,

        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        k_ml=1,
        k_val_ml=5,
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.4,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=50,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        clip_gradients=False,
        experiment_name='omniglot',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    iterations=1000
    ssml_maml_gan = SSMLMAMLGAN(

        perc=labeled_percentage,
        accessible_labels=L,
        ssml_maml=ssml_maml,

        gan=gan,
        latent_dim=latent_dim,
        generated_image_shape=shape,
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5,
        k_ml=1,
        k_val_ml=1,
        k_val=1,
        k_val_val=1,
        k_val_test=1,
        k_test=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.4,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        clip_gradients=False,
        experiment_name='omniglot_p1_0.5_shift_a',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    ssml_maml_gan.train(iterations=1000)
    ssml_maml_gan.evaluate(50, 100, seed=42)
