
import numpy as np

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.data_utils import get_file

import base
import segment
import utils

class CharacterImageGenerator(segment.TextImageGenerator):
    """A class for generating train/test data to CharacterImageModel.

    This class is inhereted from segment.TextImageGenerator,
    because it represents a special case of segment.TextImageGenerator,
    when text is generated randomly and maximum length of generated text is equal to 1.

    Note:
        Initializes data with list of pregenerated characters to create images with.
        Overrides get_data_batch method to return the pair (images with character, character).

    Args:
        batch_size (`int`, optional):           Count of character images, that is generated, when get_train/validation_data methods are called.
        train_characters (`int`, optional):     Count of character images, that is generated and used only for training a model on every epoch of train/test.
        characters_per_epoch (`int`, optional): Total count of character images, that is used for training and testing a model on every epoch of train/test.
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
        train_characters = 12800,
        characters_per_epoch = 16000,
        fonts = ['FreeMono', 'Hack', 'Lato', 'LiberationSans', 'RobotoSlab',
        'UbuntuMono', 'InputSansNarrow', 'Inconsolata'],
        font_size = 25,
        image_height = 64,
        image_width = 64,
        image_border = 4,
        max_image_noise = 0.1,
        max_image_blur = 1.0,
        min_image_contrast = 0.60,
        min_image_brightness = 0.80):

        super(CharacterImageGenerator, self).__init__(
            batch_size = batch_size,
            train_texts = train_characters,
            texts_per_epoch = characters_per_epoch,
            text_max_length = 1,
            text_from_codecs = False,
            fonts = fonts,
            font_size = font_size,
            image_height = image_height,
            image_width = image_width,
            image_border = image_border,
            max_image_noise = max_image_noise,
            max_image_blur = max_image_blur,
            min_image_contrast = min_image_contrast,
            min_image_brightness = min_image_brightness)

        self.get_data_batch = self.generate_character_image_batch

    def generate_character_image_batch(self, index, size):

        if K.image_data_format() == 'channels_first':
            character_images = np.zeros([size, 1, self.image_height, self.image_width])
        else:
            character_images = np.zeros([size, self.image_height, self.image_width, 1])

        character_masks = np.zeros([size, len(self.alphabet())])

        for i in range(size):

            character = self.data[index + i]
            if len(character) > 1:
                character = character[0]

            character_image, character_segments = utils.generate_text_image(character,
                fonts = self.fonts,
                font_size = self.font_size,
                height = self.image_height,
                width = self.image_width,
                border = self.image_border,
                max_noise = self.max_image_noise,
                max_blur = self.max_image_blur,
                min_contrast = self.min_image_contrast,
                min_brightness = self.min_image_brightness)

            character_image = character_image[:, :, character_segments[0]:character_segments[1]]
            character_image = utils.extend_image_size(character_image, new_width = self.image_width)

            c = self.alphabet().index(character)
            character_masks[i, c] = 1.0

            if K.image_data_format() == 'channels_first':
                character_images[i, 0, :, :] = character_image[0]
            else:
                character_images[i, :, :, 0] = character_image[0]

        return (character_images, character_masks)


class CharacterImageModel(base.Model):
    """A class for classifying characters on images.
    Implemented by Convolutional Neural Network architecture.

    Note:
        This class is inhereted from base.Model.
        Uses CharacterImageGenerator for generating train/test data.

    Args:
        batch_size (`int`, optional):           Count of character images, that is generated, when get_train/validation_data methods are called.
        train_characters (`int`, optional):     Count of character images, that is generated and used only for training a model on every epoch of train/test.
        characters_per_epoch (`int`, optional): Total count of character images, that is used for training and testing a model on every epoch of train/test.
        fonts (list of `st`s, optional):        Names of used fonts (used for data generation).
        font_size (`int`, optional):            Font size (used for data generation).
        image_height (`int`, optional):         Image height in pixels (used for data generation).
        image_width (`int`, optional)           Image width in pixels (used for data generation).
        image_border (`int`, optional)          Image border in pixels (used for data generation).
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
        fonts = ['FreeMono', 'Hack', 'Lato', 'LiberationSans', 'RobotoSlab'],
        font_size = 25,
        image_height = 64,
        image_width = 64,
        image_border = 4,
        max_image_noise = 0.1,
        max_image_blur = 1.0,
        min_image_contrast = 0.60,
        min_image_brightness = 0.80,
        convolution_filters_1 = 64,
        convolution_filters_2 = 32,
        kernel_size = (3, 3),
        pool_size = 2,
        hidden_layer = 256,
        droupout_rate = 0.25):

        super(CharacterImageModel, self).__init__('classify',
            output_directory = output_directory)

        self.data_generator = CharacterImageGenerator(
            batch_size = batch_size,
            train_characters = train_texts,
            characters_per_epoch = texts_per_epoch,
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
        self.hidden_layer = hidden_layer
        self.droupout_rate = droupout_rate

        self._compile()

    def _compile(self):

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
        inner_layer = Dense(self.hidden_layer, activation = 'relu')(inner_layer)
        inner_layer = Dropout(self.droupout_rate)(inner_layer)
        output_layer = Dense(len(self.data_generator.alphabet()), activation='softmax')(inner_layer)

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


    def _decode_output(self, character_masks, array):

        alphabet = self.data_generator.alphabet()

        characters = []
        for character_mask in character_masks:
            c = np.argmax(character_mask)
            characters.append(alphabet[c])

        if not array:
            characters = characters[0]

        return characters

    def predict(self, images):

        images, array = self._encode_input(images)
        character_masks = self.backing_model.predict(images)
        characters = self._decode_output(character_masks, array)

        return characters
