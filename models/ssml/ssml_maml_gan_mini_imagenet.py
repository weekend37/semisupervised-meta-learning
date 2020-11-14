import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers

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
from models.ssml.ssml_maml_gan import SSMLMAML, SSMLMAMLGAN

if __name__ == '__main__':
    mini_imagenet_database = MiniImagenetDatabase()
    shape = (84, 84, 3)
    latent_dim = 512
    mini_imagenet_generator = get_generator(latent_dim)
    mini_imagenet_discriminator = get_discriminator()
    mini_imagenet_parser = MiniImagenetParser(shape=shape)
    labeled_percentage = 0.5

    # Here we want to choose what labels we have access to through out this wohle process. 
    # if L is the dictionary containing this information then maybe something like:
    # (see get_supervised_meta_learning_dataset() in ssml_maml_gan for filesystem navigation and label handling)
    # L = {}
    # for f in folders:
    #   all_images = os.GET_ALL_IMAGES_IN_THAT_FOLDER # not sure what the syntax is
    #   L[f] = np.random.choose(all_images)
    L = None

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

    GAN_EPOCHS = 1
    print("Training GAN for", GAN_EPOCHS,"epochs")
    gan.perform_training(epochs=GAN_EPOCHS, checkpoint_freq=min(5,GAN_EPOCHS))
    gan.load_latest_checkpoint()

    print("training GAN is done")
    time.sleep(1)

    ssml_maml = SSMLMAML(
        
        perc=labeled_percentage,
        accessible_labels = L, 

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

        perc=labeled_percentage,
        accessible_labels = L, 
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

    ssml_maml_gan.visualize_meta_learning_task(shape, num_tasks_to_visualize=2, perc=labeled_percentage)

    ssml_maml_gan.train(iterations=10000)
    ssml_maml_gan.evaluate(10, 100, seed=42)

