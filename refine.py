
import numpy as np

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.recurrent import GRU
from keras.layers.merge import concatenate

import base

class SegmentPredictionGenerator(base.Generator):
    """A class for generating train/test data to RefineSegmentPredictionModel.

    This class is inhereted from base.Generator.

    Note:
        Not initializes data.
        Overrides get_data_batch method to return the pair (predicted segments, segments).

    Args:
        segment_model (segment.TextImageModel):   Model to refine the predictions of.
        encode_func (function (list of `int`s)):  Function to encode output of model to refine
                                                  to input of refining model.          

    Attributes:
        -

    """

    def __init__(self,
        segment_model,
        encode_func):
        
        generator = segment_model.data_generator

        super(SegmentPredictionGenerator, self).__init__(
            batch_size = generator.batch_size, 
            train_data_size = generator.train_data_size,
            data_size = generator.data_size)
        
        self.segment_model = segment_model
        self.encode = encode_func
        self.get_data_batch = self.get_segment_prediction_batch

    def get_segment_prediction_batch(self, index, size):
        text_images, segment_masks = self.segment_model.data_generator.get_text_image_batch(index, size)
        predicted_segment_masks = self.segment_model.predict(text_images)
        return (self.encode(predicted_segment_masks), segment_masks)

class SegmentPredictionModel(base.Model):
    """A class for refining prediction on segments of SegmentTextImageModel.
    Implemented by Recurrent Neural Network architecture.

    Note:
        This class is inhereted from BaseModel.
        Uses SegmentPredictionGenerator for generating train/test data.

    Args:
        segment_text_image_model (segment.TextImageModel): Model to refine the predictions of.
        output_directory (`str`, optional):                Count of character images, that is generated and used only for training a model on every epoch of train/test.
        sequence_length (`int`, optional):                 Lenght of sequence in Input layer.
        gru_size (`int`, optional):                        Numer of units in GRU layer.
        droupout_rate ('float', optional)                  Drop fraction in 'keras' Dropout layer.

    Attributes:
        -

    """

    def __init__(self,
        segment_model,
        output_directory = 'OCR',
        sequence_length = 10,
        gru_size = 125,
        dropout_rate = 0.25):

        super(SegmentPredictionModel, self).__init__('refine',
        output_directory = output_directory)

        def encode_func(segment_masks):
            sequences, _ = self._encode_input(segment_masks)
            return sequences

        self.data_generator = SegmentPredictionGenerator(segment_model, encode_func)

        self.sequence_length = sequence_length
        self.gru_size = gru_size
        self.dropout_rate = dropout_rate

        self.image_width = segment_model.image_width
        self.image_height = segment_model.image_height

        self._compile()

    def _compile(self):

        input_shape = (self.image_width  - self.sequence_length + 1, self.sequence_length)
        input_layer = Input(shape = input_shape, dtype = 'float32')
        
        inner_layer_1 = GRU(self.gru_size, return_sequences = True)(input_layer)
        inner_layer_2 = GRU(self.gru_size, return_sequences = True, go_backwards = True)(input_layer)
        inner_layer = Flatten()(concatenate([inner_layer_1, inner_layer_2]))
        inner_layer = Dropout(self.dropout_rate)(inner_layer)
        
        output_layer = Dense(self.image_width, activation = 'sigmoid')(inner_layer)

        self.backing_model = Model(inputs = input_layer, outputs = output_layer)
        self.backing_model.compile(loss = keras.losses.binary_crossentropy,
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])

    def _encode_input(self, segment_masks):

        segment_masks = np.array(segment_masks)
        segment_mask_dims = len(segment_masks.shape)

        array = segment_mask_dims is not 1
        if not array:
            segment_masks = np.expand_dims(segment_masks, 0)

        sequences = []
        for segment_mask in segment_masks:
            sequence = []
            for x in range(len(segment_mask) - self.sequence_length + 1):
                sequence.append(segment_mask[x : x + self.sequence_length])
            sequences.append(np.array(sequence))

        return np.array(sequences), array

    def _decode_output(self, refined_segment_masks, array):

        if not array:
            refined_segment_masks = refined_segment_masks[0]

        return refined_segment_masks

    def predict(self, segment_masks):

        sequences, array = self._encode_input(segment_masks)
        refined_segment_masks = self.backing_model.predict(sequences)
        refined_segment_masks = self._decode_output(refined_segment_masks, array)

        return refined_segment_masks