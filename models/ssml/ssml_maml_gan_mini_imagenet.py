import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers

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
from models.ssml.ssml_maml_gan import SSMLMAML, SSMLMAMLGAN

if __name__ == '__main__':
    mini_imagenet_database = MiniImagenetDatabase()
    shape = (84, 84, 3)
    latent_dim = 512
    mini_imagenet_generator = get_generator(latent_dim)
    mini_imagenet_discriminator = get_discriminator()
    mini_imagenet_parser = MiniImagenetParser(shape=shape)

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
    gan.perform_training(epochs=1, checkpoint_freq=5)
    gan.load_latest_checkpoint()

    print("-------------------------here--------------------------")
    
    ssml_maml = SSMLMAML(
        
        perc=0.5,

        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k_ml=1,
        k_val_ml=1,
        k_val=1,
        k_val_val=1,
        k_test=1,
        k_val_test=1,
        meta_batch_size=4,
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

        perc=0.5,
        ssml_maml=ssml_maml,

        gan=gan,
        latent_dim=latent_dim,
        generated_image_shape=shape,
        database=mini_imagenet_database,
        network_cls=MiniImagenetModel,
        n=5,
        k_ml=1,
        k_val=1,
        k_val_ml=1,
        k_val_val=1,
        k_val_test=1,
        k_test=1,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=0.001,
        report_validation_frequency=200,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        clip_gradients=False,
        experiment_name='mini_imagenet_p3',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )

    ssml_maml_gan.visualize_meta_learning_task(shape, num_tasks_to_visualize=2, perc=0.5)

    ssml_maml_gan.train(iterations=15000)
    ssml_maml_gan.evaluate(50, seed=42)

