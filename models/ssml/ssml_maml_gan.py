from tensorflow import keras
from tensorflow.keras import layers

from models.lasiummamlgan.maml_gan import MAMLGAN
from models.maml.maml import ModelAgnosticMetaLearningModel

# TODO: add imports

maml = ModelAgnosticMetaLearningModel(
    database=mini_imagenet_database,
    network_cls=MiniImagenetModel,
    n=5,
    k_ml=1,
    k_val_ml=5,
    k_val=1,
    k_val_val=15,
    k_test=15,
    k_val_test=15,
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
    val_test_batch_norm_momentum=0.0,
)


class SSMLMAMLGAN(MAMLGAN):
    def __init__(self, perc, *args, **kwargs):
        super(SSMLMAMLGAN, self).__init__(*args, **kwargs)
        self.perc = perc

    def merge_train_dataset(self):
        maml_gan_ds = self.get_train_dataset()
        maml_ds = maml.get_train_dataset()
        return maml_gan_ds, maml_ds

    def train(self, iterations=5):
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)
        maml_gan_train_dataset, maml_train_dataset = self.merge_train_dataset()
        iteration_count = self.load_model()
        epoch_count = iteration_count // tf.data.experimental.cardinality(maml_train_dataset)
        pbar = tqdm(maml_train_dataset)

        train_accuracy_metric = tf.metrics.Mean()
        train_accuracy_metric.reset_states()
        train_loss_metric = tf.metrics.Mean()
        train_loss_metric.reset_states()

        should_continue = iteration_count < iterations
        while should_continue:
            for (train_ds, val_ds), (train_labels, val_labels) in train_dataset:
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


# class SSMLMAML(ModelAgnosticMetaLearningModel):
#     def __init__(self, perc, *args, **kwargs):
#         super(ModelAgnosticMetaLearningModel, self).__init__(*args, **kwargs)
#         self.perc = perc
#
#     def get_train_dataset(self):
#         # MAML.get_train
#         # MAMLGAN.




