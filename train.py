import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import tensorflow.keras.backend as K

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os 
import time
import argparse

from data import data_generator
from model import noise_reduction


def main(args):

    ############################################################################
    ## create folder
    ############################################################################
    create_date_time = time.strftime("%Y%m%d-%H%M%S")
    train_dir = os.path.join("./experiment", create_date_time)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok = True)
    print(f"[saving path] {train_dir}")

    ############################################################################
    ## create data generator
    ############################################################################
    train_dataset, train_dataset_count = data_generator.create_data_generator(clean_sounds_path = os.path.join(args.train_path, "train"),
                                                         noisy_sounds_path = os.path.join(args.train_path, "train"),
                                                         batch_size = args.batch_size)

    test_dataset, test_dataset_count = data_generator.create_data_generator(clean_sounds_path = os.path.join(args.test_path, "valid"),
                                                        noisy_sounds_path = os.path.join(args.test_path, "valid"),
                                                        batch_size = args.batch_size)

    ############################################################################
    ## create model
    ############################################################################
    model_wrapper = noise_reduction.NoiseReduction(args)
    model = model_wrapper.build_model(input_size = args.input_size, verbose = args.model_verbose)

    if args.pretrained_model:
        trained_model = keras.models.load_model(args.pretrained_model, 
                                                compile = False, 
                                                custom_obects = {'tf': tf})
        model.set_weights(trained_model.get_weights())
        print(f"[load model weight] load pretrained model weight from {args.pretrained_model}")


    ############################################################################
    ## training model
    ############################################################################

    from tensorflow.keras.callbacks import LearningRateScheduler
    def learning_rate(step_size, decay, verbose = 1):
        def schedule(epoch, lr):
            if epoch > 0 and epoch % step_size == 0:
                return lr * decay
            else:
                return lr
        return LearningRateScheduler(schedule, verbose = verbose)


    def tensor_board(path):
        return TensorBoard(log_dir = os.path.join(path, 'log'), 
                           histogram_freq = 0,
                           write_graph = True,
                           write_images = True,
                           update_freq = 5000, ## 'epoch' for save every epoch 
                           embeddings_freq = 0, ## 1 for enable 
                           )

    callbacks = [
        learning_rate(step_size = args.learning_rate_step_size, 
                      decay = args.learning_rate_decay),
        tensor_board(train_dir),
    ]


    model.compile(optimizer = tf.keras.optimizers.Adam(lr = args.learning_rate),
                  loss = tf.keras.losses.MeanAbsoluteError())

    history = model.fit(train_dataset,
                        epochs = args.epochs, #500,
                        batch_size = args.batch_size, #16, 
                        validation_data = test_dataset,
                        validation_steps = test_dataset_count, #5078,
                        callbacks = callbacks,
                        shuffle = True, 
                        )

    ############################################################################
    ## evaluate
    ############################################################################

    print("[model evaluate]")
    model.evaluate(test_dataset)
    model.save(os.path.join(train_dir, "NoiseSuppressionModel.h5"))


    ############################################################################
    ## convert to coreml
    ############################################################################

    model_wrapper.create_mlmodel(os.path.join(train_dir, "NoiseSuppressionModel.mlmodel"))


def parser():
    parser = argparse.ArgumentParser(description='speech denoise')

    # --------------
    #  Training
    # --------------
    parser.add_argument('--epochs',
                        type = int,
                        default = 500,
                        help = 'number of epoch to train')

    parser.add_argument('--batch-size',
                        type = int,
                        default = 16,
                        help = 'number ofbatch size')

    parser.add_argument('--learning-rate',
                        type = float,
                        default = 1e-3,
                        help = 'learning rate')

    parser.add_argument('--learning-rate-step-size',
                        type = int,
                        default = 50,
                        help = 'learning rate step size in epochs')

    parser.add_argument('--learning-rate-decay',
                        type = float,
                        default = 0.9,
                        help = 'learning rate decay at each step')
    # --------------
    #  Data
    # --------------
    parser.add_argument('--train-path',
                        type = str,
                        required = True,
                        help = 'audio train directory')
    parser.add_argument('--test-path',
                        type = str,
                        required = True,
                        help = 'audio test directory')
    # --------------
    #  Model
    # --------------
    parser.add_argument('--input-size',
                        type = int,
                        default = 12000,
                        help = 'wave input size')

    parser.add_argument('--kernel-size',
                        type = int,
                        default = 32,
                        help = 'convolution kernel size')

    parser.add_argument('--strides',
                        type = int,
                        default = 2,
                        help = 'convolution kernel strides')

    parser.add_argument('--model-verbose',
                        action = 'store_true',
                        help = 'verbose model summary')
    # --------------
    #  Pretrain model
    # --------------
    parser.add_argument('--pretrained-model',
                        type = str,
                        help = 'pretrain model path')
    # --------------
    #  Hardware
    # --------------
    parser.add_argument('--gpu-memory-fraction',
                        type = float,
                        default=0.8,
                        help = 'fraction of GPU memory to allocate')

    parser.add_argument('--use-cpu',
                        action = 'store_true',
                        help = 'force tensorflow use cpu')

    return parser


def init_session(gpu_memory_fraction):
    if int(tf.__version__.split(".")[0]) < 2:
        keras.backend.set_session(tensorflow_session(gpu_memory_fraction = gpu_memory_fraction))

def tensorflow_session(gpu_memory_fraction):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    return tf.Session(config = config)


if __name__ == '__main__':
    args = parser().parse_args()

   if args.use_cpu:
        cpu_devices = tf.config.experimental.list_physical_devices(device_type = 'CPU')
        tf.config.experimental.set_visible_devices(devices = cpu_devices, device_type = 'CPU')
        tf.debugging.set_log_device_placement(True)

    init_session(args.gpu_memory_fraction)
    main(args)








