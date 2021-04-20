from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
import tensorflow as tf
import coremltools

import glob
from tqdm import tqdm
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import time

################################################################################
### load data
################################################################################

clean_sounds = glob.glob('/Volumes/IPEVO_X0244/speech_to_text/speech_to_text_dataset/facebookresearch/train/clean_trainset_wav/*')
noisy_sounds = glob.glob('/Volumes/IPEVO_X0244/speech_to_text/speech_to_text_dataset/facebookresearch/train/noisy_trainset_wav/*')

clean_sounds_list, _ = tf.audio.decode_wav(tf.io.read_file(clean_sounds[0]), desired_channels = 1)
for i in tqdm(clean_sounds[1:]):
    so, _ = tf.audio.decode_wav(tf.io.read_file(i), desired_channels = 1)
    clean_sounds_list = tf.concat((clean_sounds_list, so), 0)

noisy_sounds_list, _ = tf.audio.decode_wav(tf.io.read_file(noisy_sounds[0]), desired_channels = 1)
for i in tqdm(noisy_sounds[1:]):
    so, _ = tf.audio.decode_wav(tf.io.read_file(i), desired_channels = 1)
    noisy_sounds_list = tf.concat((noisy_sounds_list, so), 0)

print(f"clean sound list : {clean_sounds_list.shape}, noisy sounds list : {noisy_sounds_list.shape}")
# clean_sounds_list = clean_sounds_list[: len(clean_sounds_list)]
# noisy_sounds_list = noisy_sounds_list[: len(noisy_sounds_list)]

clean_sounds_list = clean_sounds_list[: len(clean_sounds_list) // 2]
noisy_sounds_list = noisy_sounds_list[: len(noisy_sounds_list) // 2]

batching_size = 12000
clean_train,noisy_train = [], []

#{##############################################################################
urban_sound_list = glob.glob("/Volumes/IPEVO_X0244/speech_to_text/speech_to_text_dataset/UrbanSound8K/audio/fold*/*")
concat_len = clean_sounds_list.shape[0]
noise_concat = np.array([0]).reshape((1,1))
while noise_concat.shape[0]  < concat_len + 1:
    try:
        file = np.random.choice(urban_sound_list)
        wave, _ = tf.audio.decode_wav(tf.io.read_file(file), desired_channels = 1)
        noise_concat = np.concatenate((noise_concat, wave))
    except:
        pass
noise_concat = noise_concat[1: 1 + concat_len]

clean_sounds_list = tf.concat((clean_sounds_list, clean_sounds_list), 0)
noisy_sounds_list = tf.concat((noisy_sounds_list, noise_concat), 0)
#}##############################################################################

for i in tqdm(range(0, clean_sounds_list.shape[0] - batching_size, batching_size)):
    clean_train.append(clean_sounds_list[i: i + batching_size])
    noisy_train.append(noisy_sounds_list[i: i + batching_size])

clean_train = tf.stack(clean_train)
noisy_train = tf.stack(noisy_train)

print(f"clean train shape: {clean_train.shape}")
print(f"noise train shape: {noisy_train.shape}")
del clean_sounds_list, noisy_sounds_list

################################################################################
## create a tf.data.Dataset
################################################################################

def get_dataset(x_train, y_train):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(100).batch(64, drop_remainder = True)
    return dataset

split_factor = 0.8
split_size = int(noisy_train.shape[0] * split_factor)
train_dataset = get_dataset(noisy_train[:split_size], clean_train[:split_size])
test_dataset = get_dataset(noisy_train[split_size:], clean_train[split_size:])
# train_dataset = get_dataset(noisy_train[:40000], clean_train[:40000])
# test_dataset = get_dataset(noisy_train[40000:], clean_train[40000:])
del clean_train, noisy_train

################################################################################
## create model
################################################################################

SCALE = 2

def build_model(input_size = 64000):
    input_shape = layers.Input(shape = (input_size, 1))
    x= input_shape
    if input_size == 4410:
        paddings = tf.constant([[0, 0], [0, 6], [0, 0]])        
        x = tf.pad(x, paddings)
        
    c1 = layers.Conv1D(filters = 2 * SCALE, 
                       kernel_size = 32,
                       strides = 2,
                       padding = 'same',
                       activation = 'relu')(x)
    c2 = layers.Conv1D(filters = 4 * SCALE,
                       kernel_size = 32,
                       strides = 2,
                       padding = 'same',
                       activation = 'relu')(c1)
    c3 = layers.Conv1D(filters = 8 * SCALE,
                       kernel_size = 32,
                       strides = 2,
                       padding = 'same',
                       activation = 'relu')(c2)
    c4 = layers.Conv1D(filters = 16 * SCALE,
                       kernel_size = 32,
                       strides = 2,
                       padding = 'same',
                       activation = 'relu')(c3)
    c5 = layers.Conv1D(filters = 32 * SCALE,
                       kernel_size = 32,
                       strides = 2,
                       padding = 'same',
                       activation = 'relu')(c4)
    dc1 = layers.Conv1DTranspose(filters = 32 * SCALE,
                                 kernel_size = 32,
                                 strides = 1,
                                 padding = 'same')(c5)
    conc = layers.Concatenate()([c5,dc1])
    dc2 = layers.Conv1DTranspose(filters = 16 * SCALE,
                                 kernel_size = 32,
                                 strides = 2,
                                 padding = 'same')(conc)
    conc = layers.Concatenate()([c4,dc2])
    dc3 = layers.Conv1DTranspose(filters = 8 * SCALE,
                                 kernel_size = 32,
                                 strides = 2,
                                 padding = 'same')(conc)
    conc = layers.Concatenate()([c3,dc3])
    dc4 = layers.Conv1DTranspose(filters = 4 * SCALE,
                                 kernel_size = 32,
                                 strides = 2,
                                 padding = 'same')(conc)
    conc = layers.Concatenate()([c2,dc4])
    dc5 = layers.Conv1DTranspose(filters = 2 * SCALE,
                                 kernel_size = 32,
                                 strides = 2,
                                 padding = 'same')(conc)
    conc = layers.Concatenate()([c1,dc5])
    dc6 = layers.Conv1DTranspose(filters = 1,
                                 kernel_size = 32,
                                 strides = 2,
                                 padding = 'same')(conc)
    conc = layers.Concatenate()([x,dc6])
    dc7 = layers.Conv1DTranspose(filters = 1,
                                 kernel_size = 32,
                                 strides = 1,
                                 padding = 'same',
                                 activation = 'linear')(conc)
    if input_size == 4410:
        dc7 = dc7[:, :input_size, :]
    model = models.Model(inputs = input_shape, outputs = dc7)
    return model
model = build_model(input_size = batching_size)
#model.summary()
# model.load_weights("./experiment/20210222-185036/NoiseSuppressionModel.h5")

################################################################################
## create folder
################################################################################
create_date_time = time.strftime("%Y%m%d-%H%M%S")
train_dir = os.path.join("./experiment", create_date_time)
if not os.path.exists(train_dir):
    os.makedirs(train_dir, exist_ok = True)

################################################################################
## training model
################################################################################

from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
def learning_rate(step_size, decay, verbose = 1):
    def schedule(epoch, lr):
        if epoch > 0 and epoch % step_size == 0:
            return lr * decay
        else:
            return lr
    return LearningRateScheduler(schedule, verbose = verbose)


def tensor_board(path):
    return TensorBoard(log_dir = os.path.join(path, 'log'), 
                       histogram_freq = 1,
                       write_graph = True,
                       write_images = False,
                       update_freq = 5000, ## 'epoch' for save every epoch 
                       # profile_batch = 0,  ## 2 for enable
                       embeddings_freq = 0, ## 1 for enable 
                       )

callbacks = [
    learning_rate(step_size = 100, decay = 0.9),
    tensor_board(train_dir),
]
# lr = 2e-3
lr = 3e-3
model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=tf.keras.losses.MeanAbsoluteError())
history = model.fit(train_dataset,
                    epochs = 1000,
                    batch_size = 16, 
                    validation_data = test_dataset,
                    validation_steps = test_dataset.__len__(),
                    callbacks = callbacks,
                    shuffle = True, 
                    )

################################################################################
## evaluate
################################################################################

print("[model evaluate]")
model.evaluate(test_dataset)
model.save(os.path.join(train_dir, "NoiseSuppressionModel.h5"))

################################################################################
## inference
################################################################################

# def get_audio(path):
#   audio,_ = tf.audio.decode_wav(tf.io.read_file(path),1)
#   return audio

# def inference_preprocess(path):
#   audio = get_audio(path)
#   audio_len = audio.shape[0]
#   batches = []
#   for i in range(0,audio_len-batching_size,batching_size):
#     batches.append(audio[i:i+batching_size])

#   batches.append(audio[-batching_size:])
#   diff = audio_len - (i + batching_size)
#   return tf.stack(batches), diff

# def predict(path):
#   test_data,diff = inference_preprocess(path)
#   predictions = model.predict(test_data)
#   final_op = tf.reshape(predictions[:-1],((predictions.shape[0]-1)*predictions.shape[1],1))
#   final_op = tf.concat((final_op,predictions[-1][-diff:]),axis=0)
#   return final_op

# fig = plt.Figure()
# librosa.display.waveplot(np.squeeze(get_audio(noisy_sounds[4]).numpy(),-1))
# librosa.display.waveplot(np.squeeze(predict(noisy_sounds[4])))
# fig.savefig('inference.png')


################################################################################
## convert to coreml
################################################################################

converted_model = build_model(input_size = 4410)
converted_model.set_weights(model.get_weights())
mlmodel = coremltools.convert(converted_model)
mlmodel.save(os.path.join(train_dir, "NoiseSuppressionModel.mlmodel"))







