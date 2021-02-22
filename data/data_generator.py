import os
import numpy as np
import random
import tensorflow as tf



def get_dataset(x_train,y_train, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(100).batch(batch_size, drop_remainder = True)
    return dataset

def create_data_generator(
    clean_sounds_path,
    noisy_sounds_path,
    batch_size = 64
):

    clean_sounds = glob.glob(os.path.join(clean_sounds_path, '*'))
    noisy_sounds = glob.glob(os.path.join(noisy_sounds_path, '*'))

    clean_sounds_list, _ = tf.audio.decode_wav(tf.io.read_file(clean_sounds[0]),desired_channels = 1)
    for i in tqdm(clean_sounds[1:]):
        so,_ = tf.audio.decode_wav(tf.io.read_file(i), desired_channels = 1)
        clean_sounds_list = tf.concat((clean_sounds_list,so) , 0)

    noisy_sounds_list, _ = tf.audio.decode_wav(tf.io.read_file(noisy_sounds[0]), desired_channels = 1)
    for i in tqdm(noisy_sounds[1:]):
        so,_ = tf.audio.decode_wav(tf.io.read_file(i), desired_channels = 1)
        noisy_sounds_list = tf.concat((noisy_sounds_list, so), 0)

    print(f"clean sound list : {clean_sounds_list.shape}, noisy sounds list : {noisy_sounds_list.shape}")

    batching_size = 12000
    clean_train, noisy_train = [], []

    for i in tqdm(range(0, clean_sounds_list.shape[0] - batching_size, batching_size)):
        clean_train.append(clean_sounds_list[i: i + batching_size])
        noisy_train.append(noisy_sounds_list[i: i + batching_size])

    clean_train = tf.stack(clean_train)
    noisy_train = tf.stack(noisy_train)

    print(f"clean train shape: {clean_train.shape}")
    print(f"noise train shape: {noisy_train.shape}")

    ################################################################################
    ## create a tf.data.Dataset
    ################################################################################

    # train_dataset = get_dataset(noisy_train[:40000],clean_train[:40000], batch_size)
    # test_dataset = get_dataset(noisy_train[40000:],clean_train[40000:], batch_size)

    return get_dataset(noisy_train, clean_train, batch_size), len(clean_sounds_list)

