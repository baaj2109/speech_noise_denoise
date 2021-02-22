
from .model_interface import model_interface

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

import coremltools

class NoiseReduction(model_interface):
    def __init__(self, args):
        try:
            self.kernel_size = args.kernel_size # 32
            self.strides = args.strides # 2
        except:
            print("build NoiseReduction failed")

    def build_model(self, input_size = 12000, verbose = False):

        input_shape = layers.Input(shape=(input_size, 1))
        x = input_shape
        if input_size == 4410:
            paddings = tf.constant([[0,0],[0,6],[0,0]])        
            x = tf.pad(x, paddings)

        c1 = layers.Conv1D(filters = 2, 
                           kernel_size = self.kernel_size,
                           strides = self.strides,
                           padding = 'same',
                           activation = 'relu')(x)
        c2 = layers.Conv1D(filters = 4,
                           kernel_size = self.kernel_size,
                           strides = self.strides,
                           padding = 'same',
                           activation = 'relu')(c1)
        c3 = layers.Conv1D(filters = 8,
                           kernel_size = self.kernel_size,
                           strides = self.strides,
                           padding = 'same',
                           activation = 'relu')(c2)
        c4 = layers.Conv1D(filters = 16,
                           kernel_size = self.kernel_size,
                           strides = self.strides,
                           padding = 'same',
                           activation = 'relu')(c3)
        c5 = layers.Conv1D(filters = 32,
                           kernel_size = self.kernel_size,
                           strides = self.strides,
                           padding = 'same',
                           activation = 'relu')(c4)
        dc1 = layers.Conv1DTranspose(filters = 32,
                                     kernel_size = self.kernel_size,
                                     strides = 1,
                                     padding = 'same')(c5)
        conc = layers.Concatenate()([c5, dc1])
        dc2 = layers.Conv1DTranspose(filters = 16,
                                     kernel_size = self.kernel_size,
                                     strides = self.strides,
                                     padding = 'same')(conc)
        conc = layers.Concatenate()([c4, dc2])
        dc3 = layers.Conv1DTranspose(filters = 8,
                                     kernel_size = self.kernel_size,
                                     strides = self.strides,
                                     padding = 'same')(conc)
        conc = layers.Concatenate()([c3, dc3])
        dc4 = layers.Conv1DTranspose(filters = 4,
                                     kernel_size = self.kernel_size,
                                     strides = self.strides,
                                     padding = 'same')(conc)
        conc = layers.Concatenate()([c2, dc4])
        dc5 = layers.Conv1DTranspose(filters = 2,
                                     kernel_size = self.kernel_size,
                                     strides = self.strides,
                                     padding = 'same')(conc)
        conc = layers.Concatenate()([c1, dc5])
        dc6 = layers.Conv1DTranspose(filters = 1,
                                     kernel_size = self.kernel_size,
                                     strides = self.strides,
                                     padding = 'same')(conc)
        conc = layers.Concatenate()([x,dc6])
        dc7 = layers.Conv1DTranspose(filters = 1,
                                     kernel_size = self.kernel_size,
                                     strides = 1,
                                     padding = 'same',
                                     activation = 'linear')(conc)
        if input_size == 4410:
            dc7 = dc7[:, :input_size, :]

        self.keras_model = models.Model(inputs = input_shape, outputs = dc7)

        if verbose:
            self.keras_model.summary()
        return self.keras_model


    def create_mlmodel(self, output_path = "./tmp.mlmodel"):

        try:
            converted_model = build_model(input_size = 4410)
            converted_model.set_weights(model.get_weights())
            mlmodel = coremltools.convert(model)

            if not os.path.exists(output_path.rsplit("/",1)[0]):
                os.makedirs(os.path.join(output_path.rsplit("/",1)[0]))

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    mlmodel.save(output_path)
                    print("create mlmodel complete.")
                except Warning as e:
                    print(f"failed to convert to CoreML with error: {e}")
        except:
            print("convert to CoreML model failed.")


def parser():
    parser = argparse.ArgumentParser(description='waveform u net')

    parser.add_argument('--convert',
                        action='store_true',
                        help='convert h5 to mlmodel')
    parser.add_argument('--model',
                        type = str,
                        help = "keras model path: ./model.h5")
    parser.add_argument('--output',
                        type = str,
                        help = "CoreML model path: ./model.mlmodel")
    return parser

if __name__ == '__main__':

    denoise_model = NoiseReduction()
    print(denoise_model)

