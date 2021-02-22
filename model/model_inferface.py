
import tensorflow as tf


class model_interface:

    def __init__():
        raise NotImplementedError("init")

    def build_model(self, input_size, verbose):
        raise NotImplementedError("build_model")

    def load_model(self, model_path):
        try:
            self.keras_model = load_model(model_path, compile = False, custom_objects = {'tf':tf})
            print("load model complete.")
        except:
            print("load model failed.")        
        # raise NotImplementedError("load_model")

    def save_model(self, model_path = "./tmp.h5"):
        if not os.path.exists(output_path.rsplit("/",1)[0]):
            os.mkdirs(os.path.join(output_path.rsplit("/",1)[0]))
        try:
            self.keras_model.save(output_path)
            print("save model complete.")
        except:
            print("save model failed.")

        # raise NotImplementedError("save_model")

    def create_mlmodel(self, output_path = "./tmp.mlmodel"):
        raise NotImplementedError("create_mlmodel")