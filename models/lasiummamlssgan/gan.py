import os

import tensorflow as tf
import numpy as np

import settings

np.random.seed(94305)
tf.random.set_seed(94305)
class CheckPointFreq(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, epochs, freq=1, *args, **kwargs):
        super(CheckPointFreq, self).__init__(*args, **kwargs)
        self.freq = freq
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch != 0 and (epoch + 1) % self.freq == 0:
            super(CheckPointFreq, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        self.epochs_since_last_save = np.inf
        self._save_model(self.epochs, logs)

        super(CheckPointFreq, self).on_train_end(logs)


class VisualizationCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, visualization_freq=1, *args, **kwargs):
        super(VisualizationCallback, self).__init__(*args, **kwargs)
        self.visualization_freq = visualization_freq

    def on_epoch_end(self, epoch, logs=None):
        super(VisualizationCallback, self).on_epoch_end(epoch, logs)
        if epoch != 0 and epoch % self.visualization_freq == 0:
            gan = self.model
            image = gan.generator(tf.random.normal(shape=(5, self.model.latent_dim)))

            # writer = self._get_writer(self._train_run_name)
            # with writer.as_default():
                # tf.summary.image(name='x', data=image, step=epoch, max_outputs=5)


class GAN(tf.keras.models.Model):

    # this is actually the ssgan since it is inside the folder lasiummamlssgan. :P
    def __init__(
            self,
            gan_name,
            SS,
            image_shape,
            latent_dim,
            database,
            parser,
            generator,
            discriminator,
            g_learning_rate,
            d_learning_rate,
            visualization_freq,
            **kwargs
    ):
        super(GAN, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.database = database
        self.parser = parser
        self.visualization_freq = visualization_freq
        self.image_shape = image_shape
        self.gan_name = gan_name
        self.discriminator = discriminator
        self.d_optimizer = None
        self.d_learning_rate = d_learning_rate
        self.generator = generator
        self.g_optimizer = None
        self.g_learning_rate = g_learning_rate

        self.loss_fn = None

        self.SS = SS

        self.d_loss_metric = tf.keras.metrics.Mean()
        self.g_loss_metric = tf.keras.metrics.Mean()

    def compile(self, d_optimizer, g_optimizer, loss_fn, ss_loss_fn=None):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.ss_loss_fn = ss_loss_fn

    def get_gan_name(self):
        return self.gan_name

    def generate(self, z):
        return self.generator(z)

    def gan_regularization_loss(self, z1, z2, i1, i2):
        images_norm = tf.reduce_mean(tf.abs(i1 - i2), axis=[1, 2, 3])
        zs_norm = tf.reduce_mean(tf.abs(z1 - z2), axis=1)

        lz = tf.reduce_mean(images_norm / zs_norm)
        eps = 1e-5
        lz = 1 / (lz + eps)
        return lz

    def train_step(self, dataset):

        """
        strategy: since we get real_images as a tuple, here is where we split 
        into 2 parts. One part will be used for supervision, other is normal.
        
        """
        # if isinstance(real_images, tuple):
        #     real_images = real_images[0]
        
        # subsect real_images into 2. 
        batch_size = tf.shape(dataset[0])[0] # total
        ## more ad-hoc way of training on 50% sup and 50% unsup.
        # this is fine since we use dataset.shuffle(reshuffle_each_iteration=False)

        # ss_labels = dataset[1][0:batch_size//2]
        # ss_images = dataset[0][0:batch_size//2]
        
        real_images = dataset[0]

        ### our code goes here...
        ## NOTE: Latest update:
        ## this is not needed since dataset shuffling is not done each iteration.
        # this indicates 0 goes to supervised, 1 goes to unsupervised.
        # indicator = dataset[2]
        # where_0 = tf.where(indicator==0)
        # where_1 = tf.where(indicator==1)
        # TODO: Some bug that creates nan loss. 
        # print(real_images.shape, ss_images.shape, ss_labels.shape)
        # ss_labels = tf.gather_nd(dataset[1],where_0)
        # ss_images = tf.gather_nd(dataset[0],where_0)
        # real_images = tf.gather_nd(dataset[0],where_1)
        ###

        batch_size = tf.shape(real_images)[0] # this becomes original batch size / 2.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        generated_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)[:,-1]
            d_loss = self.loss_fn(labels, predictions)

            # add the supervised loss
            if self.SS:    
                ss_labels = dataset[1]
                ss_predictions = self.discriminator(real_images)[:,0:-1]
                d_loss += self.ss_loss_fn(ss_labels,ss_predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_latent_vectors1 = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_latent_vectors2 = tf.random.normal(shape=(batch_size, self.latent_dim))

        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            generated_images1 = self.generator(random_latent_vectors)
            generated_images2 = self.generator(random_latent_vectors2)
            predictions1 = self.discriminator(generated_images1)[:,-1]
            predictions2 = self.discriminator(generated_images2)[:,-1]

            g_loss = self.loss_fn(misleading_labels, predictions1 + predictions2) + self.gan_regularization_loss(
                random_latent_vectors1, random_latent_vectors2, generated_images1, generated_images2
            )

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

    # def get_dataset(self, partition='train'):
    #     instances = self.database.get_all_instances(partition_name=partition)
    #     train_dataset = tf.data.Dataset.from_tensor_slices(instances).shuffle(len(instances))
    #     train_dataset = train_dataset.map(self.parser.get_parse_fn())
    #     train_dataset = train_dataset.batch(128)
    #     return train_dataset

    def get_dataset(self, partition='train'):

        instances, instance_to_class, class_ids = self.database.get_all_instances(partition_name='train', with_classes=True)
        # print("GAN THINKS IT'S THIS MANY CLASSES:", len(class_ids))

        # get dataset from the above files
        dataset = tf.data.Dataset.from_tensor_slices(instances)
        dataset = dataset.map(self.get_db_process_path(self.database, instance_to_class, class_ids))
        dataset = dataset.cache()
        # shuffle just once in the beginning now.
        dataset = dataset.shuffle(buffer_size=len(instances)) #,reshuffle_each_iteration=False)
        # chunk it. drop_remainder will be useful when testing different % of sup,unsup
        dataset = dataset.batch(32,drop_remainder=True)

        return dataset

    def get_train_dataset(self):
        return self.get_dataset(partition='train')

    def get_db_process_path(self, db, instance_to_class, class_ids):
        num_classes = len(db.train_folders)

        def process_path(file_path):
            def extract_label_from_file_path(fp):
                fp = str(fp.numpy(), 'utf-8')
                label = instance_to_class[fp]
                label = class_ids[label]
                label = tf.one_hot(label, depth=num_classes)
                # I printed shape here and it said 1200 by default.
                return label

            label = tf.py_function(extract_label_from_file_path, inp=[file_path], Tout=tf.float32)
            image = self.parser.get_parse_fn()(file_path)
            return image, label #, np.random.choice(2,1,p=[0.5,0.5])[0]
            # the last np.random.choice acts as split for sup, unsup split.
            # TODO: NOTE; change the argument p to adjust split.
            # 0.7,0.3 means more sup data.

        return process_path


    def load_latest_checkpoint(self, epoch_to_load_from=None):
        latest_checkpoint = tf.train.latest_checkpoint(
            os.path.join(
                settings.PROJECT_ROOT_ADDRESS,
                'models',
                'lasiummamlssgan', # diff 2. reflecting change below.
                'gan',
                self.get_gan_name(),
                'gan_checkpoints'
            )
        )

        if latest_checkpoint is not None:
            self.load_weights(latest_checkpoint)
            epoch = int(latest_checkpoint[latest_checkpoint.rfind('_') + 1:])
            print(f"Loaded a GAN trained for {epoch} epochs")
            return epoch
        else:
            print(f"No GAN found with name {self.get_gan_name()} .")

        return -1

    def perform_training(self, epochs, checkpoint_freq=100, vis_callback_cls=None, load_ckpt=True):

        if load_ckpt:
            initial_epoch = self.load_latest_checkpoint()

            if initial_epoch != -1:
                print(f'Continue training from epoch {initial_epoch}.')
        
        else:
            initial_epoch = -1
            print("Training GAN from scratch (set load_ckpt=False to change this)")

        train_dataset = self.get_train_dataset()

        checkpoint_callback = CheckPointFreq(
            freq=checkpoint_freq,
            filepath=os.path.join(
                settings.PROJECT_ROOT_ADDRESS,
                'models',
                'lasiummamlssgan', # diff #1.
                'gan',
                self.get_gan_name(),
                'gan_checkpoints',
                'gan_{epoch:02d}'
            ),
            save_freq='epoch',
            save_weights_only=True,
            epochs=epochs - 1
        )
        if vis_callback_cls is None:
            vis_callback_cls = VisualizationCallback

        tensorboard_callback = vis_callback_cls(
            log_dir=os.path.join(
                settings.PROJECT_ROOT_ADDRESS,
                'models',
                'lasiummamlssgan', # diff 3
                'gan',
                self.get_gan_name(),
                'gan_logs'
            ),
            visualization_freq=self.visualization_freq
        )

        callbacks = [tensorboard_callback, checkpoint_callback]

        ss_loss_fn = lambda labels,logits: \
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels,logits))

        self.compile(
            d_optimizer=tf.keras.optimizers.Adam(self.d_learning_rate),
            g_optimizer=tf.keras.optimizers.Adam(self.g_learning_rate),
            loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            ss_loss_fn=ss_loss_fn,
        )
        self.fit(
            train_dataset,
            epochs=epochs,
            callbacks=callbacks,
            initial_epoch=initial_epoch
        )
