
import os
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image

from keras.utils import plot_model

import utils

class Generator:
    """A basic class for all data generators in this package.

    This class introduces common for all generators functionality,
    such as generating train/validation data.

    Motivation for such classes is to save RAM and generate train/validation data only by request,
    then leaving this data to be released by Python GC.

    Note:
        `data`, `get_data_batch` attribute should be initialized on __init__ stage of inhereting class.

    Args:
        batch_size (`int`, optional):      Count of data units, that is generated, when get_train/validation_data methods are called.
        train_data_size (`int`, optional): Count of data units, that is used only for training a model.
        data_size (`int`, optional):       Total count of data uints, that is used for training and testing a model.

    Attributes:
        data (list):                              Any data, that is pregenerated on the __init__ stage, to speed up the generation stage.
        get_data_batch (function (`int`, `int`)): Virtual-like method, that should be overriden by inhereting class.

    """

    data = []
    get_data_batch = None

    def __init__(self,
        batch_size = 32,
        train_data_size = 12800,
        data_size = 16000):

        assert train_data_size % batch_size == 0
        assert data_size % batch_size == 0
        assert train_data_size < data_size

        self.batch_size = batch_size
        self.train_data_size = train_data_size
        self.data_size = data_size

        self._reset()

    def _reset(self):
        self.train_index = 0
        self.validation_index = self.train_data_size

    def _shuffle_train_data(self):
        if self.data is None:
            return
        indices = list(range(self.train_data_size))
        np.random.shuffle(indices)
        indices += list(range(self.train_data_size, self.data_size))
        self.data = [self.data[i] for i in indices]

    def get_data_batch_size(self):
        return self.batch_size

    def get_steps_per_epoch(self):
        return self.train_data_size // self.batch_size

    def get_validation_steps(self):
        return (self.data_size - self.train_data_size) // self.batch_size

    def get_next_train_data(self):
        while 1:
            ret = self.get_data_batch(self.train_index, self.batch_size)
            self.train_index += self.batch_size
            if self.train_index >= self.train_data_size:
                self._shuffle_train_data()
                self._reset()
            yield ret

    def get_next_validation_data(self):
        while 1:
            ret = self.get_data_batch(self.validation_index, self.batch_size)
            self.validation_index += self.batch_size
            if self.validation_index >= self.data_size:
                self._reset()
            yield ret

class Model:
    """A basic class for all models in this package.

    This class introduces common for all models functionality,
    such as load/save weights and train/test.

    Args:
        name (`str`, optional):                             Name of this model.
        output_directory (`str`, optional):                 Path to the directory, where the training weights will be saved.
        backing_model (`keras.Model`, optional):            'Keras' package model, which this class wraps-up.
        data_generator (class inhereting BaseDataGenerator) Data generator for this model to train/test 'keras' model.

    Attributes:
        name (`str`, optional):                             Name of this model.
        backing_model (`keras.Model`, optional):            'Keras' package model, which this class wraps-up.
        data_generator (class inhereting BaseDataGenerator) Data generator for this model to train/test 'keras' model.

    """

    name = None
    backing_model = None
    data_generator = None

    def __init__(self,
        name = 'no_name',
        output_directory = 'OCR',
        backing_model = None,
        data_generator = None):

        self.name = name
        self.output_directory = output_directory
        self.backing_model = backing_model
        self.data_generator = data_generator

        self.train_history = None

    def display_architecture(self):

        output_directory = "models"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_file =  os.path.join(output_directory, self.name + ".png")
        plot_model(self.backing_model, to_file = image_file, show_shapes = True)

        return Image(filename = image_file)

    def display_train_history(self):

        if self.train_history is None:
            return

        plt.plot(self.train_history.history['acc'])
        plt.plot(self.train_history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(self.train_history.history['loss'])
        plt.plot(self.train_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def weight_file_on(self, epoch):
        return 'weights%02d.h5' % epoch

    def output_directory_for(self, run_name):
        if self.output_directory == None:
            raise RuntimeError("Output directory wasn't set!")
        return os.path.join(self.output_directory, run_name, self.name)

    def load(self, run_name, epochs):
        weight_file = os.path.join(self.output_directory_for(run_name), self.weight_file_on(epochs-1))
        self.backing_model.load_weights(weight_file)

    def train(self, run_name, start_epoch, stop_epoch):

        if start_epoch > 0:
            self.load(run_name, start_epoch)

        callbacks = []
        if self.output_directory != None:
            callbacks.append(utils.SaveModelCallback(output_directory = os.path.join(self.output_directory, run_name, self.name),
                weight_file = lambda epoch: self.weight_file_on(epoch)))

        self.display_architecture()

        self.train_history = self.backing_model.fit_generator(generator = self.data_generator.get_next_train_data(),
            steps_per_epoch = self.data_generator.get_steps_per_epoch(),
            epochs = stop_epoch,
            validation_data = self.data_generator.get_next_validation_data(),
            validation_steps = self.data_generator.get_validation_steps(),
            callbacks = callbacks,
            initial_epoch = start_epoch)
