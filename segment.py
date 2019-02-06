
import os
import re
import codecs
import numpy as np

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.data_utils import get_file

import base
import utils

class TextImageGenerator(base.Generator):
    """A class for generating train/test data to TextImageModel.

    This class is inhereted from base.Generator.

    Note:
        Initializes data with list of randomly pregenerated texts to create images with.
        Overrides get_data_batch method to return the pair (image, segments).
    
    Args:
        batch_size (`int`, optional):           Count of character images, that is generated, when get_train/validation_data methods are called.
        train_characters (`int`, optional):     Count of character images, that is generated and used only for training a model on every epoch of train/test.
        characters_per_epoch (`int`, optional): Total count of character images, that is used for training and testing a model on every epoch of train/test.
        max_text_length (`int`, optional)       Max length of text in characters.
        words_from_codecs (`bool`, optional)    The method used for generation text, codec files or random character words.
        fonts (list of `str`s, optional):       Names of used fonts.
        font_size (`int`, optional):            Font size.
        image_height (`int`, optional):         Image height in pixels.
        image_width (`int`, optional)           Image width in pixels.
        image_border (`int`, optional)          Image border in pixels.
        max_image_noise (`float`, optiinal)     Maximal gaussian-noise of the image (could be from 0.0 to 1.0).
        max_blur (`float`, optional)            Maximal blur factor (could be from 0.0 to Inf).
        min_contrast ('float', optional)        Minimal contrast factor (could be from 0.0 to 1.0).
        min_brightness ('float', optional)      Minimal brightness factor (could be from 0.0 to 1.0).

    Attributes:
        -

    """
    def __init__(self,
        batch_size = 32,
        train_texts = 12800,
        texts_per_epoch = 16000,
        text_max_length = 4,
        text_from_codecs = True,
        fonts = ['FreeMono', 'Hack', 'Lato', 'LiberationSans', 'RobotoSlab'],
        font_size = 25,
        image_height = 64,
        image_width = 128,
        image_border = 4,
        max_image_noise = 0.5,
        max_image_blur = 1.0,
        min_image_contrast = 0.50,
        min_image_brightness = 0.95):

        super(TextImageGenerator, self).__init__(batch_size, train_texts, texts_per_epoch)

        self.text_max_length = text_max_length
        self.fonts = fonts
        self.font_size = font_size
        self.image_height = image_height
        self.image_width = image_width
        self.image_border = image_border
        self.max_image_noise = max_image_noise
        self.max_image_blur = max_image_blur
        self.min_image_contrast = min_image_contrast
        self.min_image_brightness = min_image_contrast

        self.get_data_batch = self.get_text_image_batch
        if text_from_codecs:
            self.prepare_data_by_codecs()
        else:
            self.prepare_data_by_random()

    def alphabet(self):
        return u'abcdefghijklmnopqrstuvwxyz '

    def is_valid(self, text):
        return re.compile(r'^[a-z ]{1,%d}$' % self.text_max_length).match(text) != None

    def prepare_data_by_codecs(self):

        download_directory = os.path.dirname(get_file('wordlists.tgz',
        origin = 'http://www.mythic-ai.com/datasets/wordlists.tgz', untar = True))

        if self.text_max_length < 10:
            mono_fraction = 1.0
        else:
            mono_fraction = 0.5

        # monogram file is sorted by frequency in english speech
        with codecs.open(os.path.join(download_directory, 'wordlist_mono_clean.txt'), mode='r', encoding='utf-8') as monogram_file:
            for line in monogram_file:
                if len(self.data) == int(self.data_size * mono_fraction):
                    break
                word = line.rstrip().lower()
                text = word
                if self.is_valid(text):
                    self.data.append(word)

        # bigram file contains common word pairings in english speech
        with codecs.open(os.path.join(download_directory, 'wordlist_bi_clean.txt'), mode='r', encoding='utf-8') as bigram_file:
            lines = bigram_file.readlines()
            for line in lines:
                if len(self.data) == self.data_size:
                    break
                words = line.rstrip().lower().split()
                text = words[0] + ' ' + words[1]
                if self.is_valid(text):
                    self.data.append(text)

        if len(self.data) != self.data_size:
            raise IOError('Could not pull enough words from supplied monogram and bigram files.')

        interlaced_data = [''] * self.data_size
        interlaced_data[0::2] = self.data[:self.data_size // 2]
        interlaced_data[1::2] = self.data[self.data_size // 2:]
        self.data = interlaced_data

    def prepare_data_by_random(self):
        
        alphabet = self.alphabet()
        
        for i in range(self.data_size):
            
            if self.text_max_length > 1:
                text_length = np.random.randint(1, self.text_max_length)
            else:
                text_length = 1

            text = []
            for i in range(text_length):
                c = np.random.randint(0, len(alphabet))
                text.append(alphabet[c])

            self.data.append(''.join(text))

    def get_text_image_batch(self, index, size):

        if K.image_data_format() == 'channels_first':
            text_images = np.zeros([size, 1, self.image_height, self.image_width])
        else:
            text_images = np.zeros([size, self.image_height, self.image_width, 1])

        segment_masks = np.zeros([size, self.image_width])

        for i in range(size):

            text = self.data[index + i]
            try:
                text_image, segment_offsets = utils.generate_text_image(text,
                    fonts = self.fonts,
                    font_size = self.font_size,
                    height = self.image_height,
                    width = self.image_width,
                    border = self.image_border,
                    max_noise = self.max_image_noise,
                    max_blur = self.max_image_blur,
                    min_contrast = self.min_image_contrast,
                    min_brightness = self.min_image_brightness)
            except:
                text = ' '
                text_image, segment_offsets = utils.generate_text_image(text,
                    fonts = self.fonts,
                    font_size = self.font_size,
                    height = self.image_height,
                    width = self.image_width,
                    border = self.image_border,
                    max_noise = self.max_image_noise,
                    max_blur = self.max_image_blur,
                    min_contrast = self.min_image_contrast,
                    min_brightness = self.min_image_brightness)

            for x in segment_offsets:
                segment_masks[i, x] = 1.0

            if K.image_data_format() == 'channels_first':
                text_images[i, 0, :, :] = text_image[0]
            else:
                text_images[i, :, :, 0] = text_image[0]

        return (text_images, segment_masks)

class TextImageModel(base.Model):
    """A class for segmenting text images on individual character images.
    Implemented by Convolutional Neural Network architecture.

    Note:
        This class is inhereted from base.Model.
        Uses TextImageGenerator for generating train/test data.

    Args:
        output_directory (`str`, optional):     Path to the directory, where the training weights will be saved.
        batch_size (`int`, optional):           Count of text images, that is generated, when get_train/validation_data methods are called.
        train_texts (`int`, optional):          Count of text images, that is generated and used only for training a model on every epoch of train/test.
        texts_per_epoch (`int`, optional):      Total count of text images, that is used for training and testing a model on every epoch of train/test.
        text_max_length (`int`, optional):      Maximal length of text in characters.
        text_from_codecs (`bool`, optional):    The method used for generation text, codec files or random character words.
        fonts (list of `string`s, optional):    Names of used fonts (used for generation).
        font_size (`int`, optional):            Font size (used for generation).
        image_height (`int`, optional):         Image height in pixels (used for generation).
        image_width (`int`, optional)           Image width in pixels (used for generation).
        image_border (`int`, optional)          Image border in pixels (used for generation).
        max_image_noise (`float`, optiinal)     Maximal gaussian-noise of the image (could be from 0.0 to 1.0).
        max_blur (`float`, optional)            Maximal blur factor (could be from 0.0 to Inf).
        min_contrast ('float', optional)        Minimal contrast factor (could be from 0.0 to 1.0).
        min_brightness ('float', optional)      Minimal brightness factor (could be from 0.0 to 1.0).
        convolution_filters_1 (`int`, optional) Number of convolution filters on the first 'keras' Conv2D layer.
        convolution_filters_2 (`int`, optional) Number of convolution filters on the second 'keras' Conv2D layer.
        kernel_size = (`int` tuple, optional)   Kernel size of both 'keras' Conv2D layers.
        pool_size (`int`, optional)             Pool size of 'keras' boths MaxPooling2D layers.
        dense_size (`int`, optional)            Number of units in 'keras' Dense layer. 
        droupout_rate ('float')                 Drop fraction in 'keras' Dropout layer.

    Attributes:
        -

    """
    def __init__(self,
        output_directory = 'OCR',
        batch_size = 32,
        train_texts = 12800,
        texts_per_epoch = 16000,
        text_max_length = 4,
        text_from_codecs = True,
        fonts = ['FreeMono', 'Hack', 'Lato', 'LiberationSans', 'RobotoSlab'],
        font_size = 25,
        image_height = 64,
        image_width = 128,
        image_border = 4,
        max_image_noise = 0.5,
        max_image_blur = 1.0,
        min_image_contrast = 0.50,
        min_image_brightness = 0.95,
        convolution_filters_1 = 64,
        convolution_filters_2 = 32,
        kernel_size = (3, 3),
        pool_size = 2,
        dense_size = 256,
        droupout_rate = 0.25):

        super(TextImageModel, self).__init__('segment',
            output_directory = output_directory)

        self.data_generator = TextImageGenerator(
            batch_size = batch_size,
            train_texts = train_texts,
            texts_per_epoch = texts_per_epoch,
            text_max_length = text_max_length,
            text_from_codecs = text_from_codecs,
            fonts = fonts,
            font_size = font_size,
            image_height = image_height,
            image_width = image_width,
            image_border = image_border,
            max_image_noise = max_image_noise,
            max_image_blur = max_image_blur,
            min_image_contrast = min_image_contrast,
            min_image_brightness = min_image_brightness)

        self.image_height = image_height
        self.image_width = image_width

        self.convolution_filters_1 = convolution_filters_1
        self.convolution_filters_2 = convolution_filters_2
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_size = dense_size
        self.droupout_rate = droupout_rate

        self.compile()

    def compile(self):
        
        if K.image_data_format() == 'channels_first':
            input_shape = (self.image_height, self.image_width, 1)
        else:
            input_shape = (self.image_height, self.image_width, 1)

        input_layer = Input(shape = input_shape, dtype='float32')
        inner_layer = Conv2D(self.convolution_filters_1, self.kernel_size, padding = 'same',
            activation = 'relu', kernel_initializer = 'he_normal')(input_layer)
        inner_layer = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(inner_layer)
        inner_layer = Conv2D(self.convolution_filters_2, self.kernel_size, padding = 'same',
            activation = 'relu', kernel_initializer = 'he_normal')(inner_layer)
        inner_layer = MaxPooling2D(pool_size = (self.pool_size, self.pool_size))(inner_layer)
        inner_layer = Flatten()(inner_layer)
        inner_layer = Dense(self.dense_size, activation = 'relu')(inner_layer)
        inner_layer = Dropout(self.droupout_rate)(inner_layer)
        output_layer = Dense(self.image_width, activation = 'sigmoid')(inner_layer)
        
        self.backing_model = Model(inputs = input_layer, outputs = output_layer)
        self.backing_model.compile(loss = keras.losses.binary_crossentropy,
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])

    def _encode_input(self, images):

        images = np.array(images)
        image_dims = len(images.shape)

        array = image_dims is not 3
        if not array:
            images = np.expand_dims(images, 0)
        
        if K.image_data_format() == 'channels_first':
            images = np.reshape(images, (images.shape[0], 1, self.image_height, self.image_width))
        else:
            images = np.reshape(images, (images.shape[0], self.image_height, self.image_width, 1))

        return images, array


    def _decode_output(self, segment_masks, array):
        
        if not array:
            segment_masks = segment_masks[0]

        return segment_masks

    def predict(self, images):
        
        images, array = self._encode_input(images)
        segment_masks = self.backing_model.predict(images)
        segment_masks = self._decode_output(segment_masks, array)

        return segment_masks