
import numpy as np

import segment
import refine
import classify
import utils

class TextImage:
    """A class for recognizing text on images.

    Implemented by union of 3 models, introduces in ocr.classify, ocr.refine, ocr.segment:
        + CNN - for segmenting text images on char images.
        + RNN - for refining prediction on image segments.
        + CNN - for classifying chars on their individual images.

    Args:
        segment_model (`segment.TextImageModel`):        1-st trained model, that could well predict
                                                         segmentation of an image on individual character images.
        refine_model (`refine.SegmentPredictionModel`):  2-nd trained model, that could well refine prediction of
                                                         segmentation of the first model.
        classify_model (`classify.CharacterImageModel`): 3-rd trained model, that could well predict
                                                         what characters in alphabet are illustrated on images.

        character_size_hint (`int`)                      Hint on character size.
        character_treshhold (`float`)                    Used in filters of image segments.
        show_segmentation ('bool')                       Show segmentation of images, that are recognized.

    Attributes:
        -

    """

    def __init__(self, segment_model, refine_model, classify_model,
        character_size_hint = 20, character_treshhold = 0.1,
        show_segmentation = False):

        self.text_image_height = segment_model.image_height
        self.text_image_width = segment_model.image_width

        self.character_image_height = classify_model.image_height
        self.character_image_width = classify_model.image_width

        self.character_size_hint = character_size_hint
        self.character_treshhold = character_treshhold
        self.show_segmentation = show_segmentation

        self.models = (segment_model, refine_model, classify_model)

    def _segment(self, text_images):
        return self.models[0].predict(text_images)

    def _refine(self, segment_masks):
        return self.models[1].predict(segment_masks)

    def _classify(self, character_images):
        return self.models[2].predict(character_images)

    def _encode_input(self, text_images):

        text_images = np.array(text_images)
        text_image_dims = len(text_images.shape)

        array = text_image_dims is not 3
        if not array:
            text_images = np.expand_dims(text_images, 0)

        return text_images, array

    def _decode_output(self, texts, array):

        if not array:
            texts = texts[0]

        return texts

    def _convert_intermediate(self, character_images):

        extended_character_images = []
        for i, character_image in enumerate(character_images):
            extended_character_images.append(utils.extend_image_size(character_image,
                new_height = self.character_image_height, new_width = self.character_image_width))

        extended_character_images = np.array(extended_character_images)

        return extended_character_images


    def recognize(self, text_images):

        text_images, array = self._encode_input(text_images)
        text_segments = self._segment(text_images)
        text_segments = self._refine(text_segments)

        texts = []
        for i, character_segments in enumerate(text_segments):

            character_images, filtered_segments = utils.split_image_on_segments(text_images[i], character_segments,
                segment_size_hint = self.character_size_hint,
                segment_treshhold = self.character_treshhold)

            if self.show_segmentation:
                utils.show_image_with_segments(text_images[i], filtered_segments)

            character_images = self._convert_intermediate(character_images)
            characters = self._classify(character_images)
            texts.append(('').join(characters).strip())

        texts = self._decode_output(texts, array)

        return texts
