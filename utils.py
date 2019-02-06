
import os
import numpy as np
import scipy as sp
import cairocffi as cairo
import keras
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageFont, ImageEnhance, ImageDraw, ImageFilter

class SaveModelCallback(keras.callbacks.Callback):
    """A class for saving model weights to a directory on the end of every epoch in test/train cicle.

    Args:
        output_directory (`str`):        Path to a directory to save model weights.
        weight_file (function (`int`)):  Function, that builds up name of a weight file by an epoch sequence number.

    Attributes:
        -

    """

    def __init__(self, output_directory, weight_file):

        self.output_directory = output_directory
        self.weight_file_on = weight_file

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    def on_epoch_end(self, epoch, logs={}):
        weight_file = os.path.join(self.output_directory, self.weight_file_on(epoch))
        self.model.save_weights(weight_file)

def split_image_on_segments(image, segment_mask, segment_size_hint = 20, segment_treshhold = 0.1):
    """A function, that splits image on segments.

    Args:
        image:             Image to split.
        segment_mask:      Segment mask is a list of float's from 0.0 to 1.0 wich has image.shape[0] elements,
                           every element represents the probability of column to be the separator between 2 segments of the image.
        segment_size_hint: Hint on segment size. Impacts the size of the window, which is moved along the width of the image,
                           zeroing all mask elements inside of it, except the maximum element.
        segment_treshold:  Segment treshold probability. All mask elements with probability less than segment treshold will be zeroed
                           and therefore couldn't be separators.

    Returns:
        Splited image segments and new segments' offsets.

    """

    p = segment_mask.copy()

    image_width = image.shape[2]
    window_size = max(segment_size_hint // 2, 2)

    for window_off in range(image_width - window_size):
        x1, x2 = window_off, window_off + window_size
        p[x1 : x2][p[x1 : x2] < p[x1 : x2].max()] = 0

    segment_offsets = np.nonzero(p / p.max() > segment_treshhold)[0]

    image_segments = []

    for i in range(len(segment_offsets) - 1):
        x1, x2 = segment_offsets[i], segment_offsets[i+1]
        image_segments.append(image[:, :, x1:x2])

    return image_segments, segment_offsets

def generate_text_image(text,
    fonts = ['FreeMono', 'Hack', 'Lato', 'LiberationSans', 'RobotoSlab',
        'UbuntuMono', 'InputSansNarrow', 'Inconsolata'],
    font_size = 25,
    height = 64,
    width = 128,
    border = 4,
    max_noise = 0.1,
    max_blur = 1.0,
    min_contrast = 0.60,
    min_brightness = 0.80):
    """A function, that generates text image.

    Args:
        text (`str`)                        Text on image.
        fonts (list of `str`s, optional):   Names of used fonts.
        font_size (`int`, optional):        Font size.
        height (`int`, optional):           Image height in pixels.
        width (`int`, optional)             Image width in pixels.
        border (`int`, optional)            Image border in pixels.
        max_noise (`float`, optional)       Maximal noise factor (could be from 0.0 to 1.0).
        max_blur (`float`, optional)        Maximal blur factor (could be from 0.0 to Inf).
        min_contrast ('float', optional)    Minimal contrast factor (could be from 0.0 to 1.0).
        min_brightness ('float', optional)  Minimal brightness factor (could be from 0.0 to 1.0).

    Returns:
        Generated image, and character offsets.

    """
    
    image = Image.new('RGB', (width, height), 'white')

    # chose font
    font_path = 'fonts/%s-%s.ttf' % (np.random.choice(fonts), np.random.choice(["Regular", "Bold"]))
    font = ImageFont.truetype(font_path, np.random.randint(20, 35))

    text_width = sum([font.getsize(character)[0] for character in text])
    text_height = max([font.getsize(character)[1] for character in text])
    
    if text_width >= (width - 2 * border) or text_height >= (height - 2 * border):
        raise IOError('Could not fit %s string into image.' % text)

    x = np.random.randint(border, width - text_width - border)
    y = np.random.randint(border, height - border - text_height)

    draw = ImageDraw.Draw(image)
    
    character_offsets = [x]
    
    for character in text:

        draw.text((x, y), character, 'black', font)

        x += font.getsize(character)[0]
        character_offsets.append(x)
    
    del draw

    # rotate image
    image = image.convert('RGBA')
    image = image.rotate(np.random.randint(-3, 3))
    image = Image.composite(image, Image.new('RGBA', (width, height), 'white'), image)
    image = image.convert('RGB')

    # set blur
    blur = np.random.uniform(0, max_blur)
    image = image.filter(ImageFilter.GaussianBlur(blur))
    
    # set contrast
    contrast = np.random.uniform(min_contrast, 1.0)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    
    # set brightness
    brightness = np.random.uniform(min_brightness, 1)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    
    # convert to grayscale
    image = ImageOps.grayscale(image)

    image = np.array(image)
    image = image.astype(np.float32) / 255
    image = np.expand_dims(image, axis = 0)

    # add noise to background
    noise = np.random.uniform(0, max_noise)
    image = image + sp.ndimage.gaussian_filter(np.random.randn(*image.shape) * noise, 1)
    image[image > 1], image[image <= 0] = 1, 0

    return image, character_offsets

def show_image_with_segments(image, segments, mask = False, color = 'b'):
    """A function, that shows an image with segments on a plot.

    Args:
        image:     Image to show.
        segments:  Segments to show on the same plot.
        mask:      Reports, whether segments is in form of mask (= True) or offsets (= False).
        color:     Color to draw segments with.

    Returns:
        -

    """
    height, width = image.shape[1], image.shape[2]
    if mask:
        x, y = np.arange(width), np.array(segments) * height / max(segments)
        plt.plot(x, y, color + '--')
    else:
        for segment in segments:
            x, y = np.repeat(segment, height), np.arange(height)
            plt.plot(x, y)

    plt.imshow(image[0], cmap = 'gray', vmin = 0.0, vmax = 1.0)
    plt.show()

def extend_image_size(image, new_height = None, new_width = None,
    in_channels_first = True, out_channels_first = True):
    """A function, that creates an extended image,
    copying the content of an input image right in the middle.

    Args:
        image:               Image to extend.
        new_height:          Height of the extended image.
        new_width:           Width of the extended image.
        in_channels_first:   Reports, whether in the shape of the input image the first element is a color channel.
        out_channels_first:  Reports, whether in the shape of the extended iamge the first element is a color channel.

    Returns:
        Extended image.

    """

    if in_channels_first:
        channels, height, width = image.shape[0], image.shape[1], image.shape[2]
    else:
        height, width, channels = image.shape[0], image.shape[1], image.shape[2]

    if not new_height:
        new_height = height

    if new_height < height:
        raise IOError("Can't extend image: new height %d is less, than image height %d!"
            % (new_height, height))

    y1 = new_height // 2 - height // 2
    y2 = y1 + height

    if not new_width:
        new_width = width

    if new_width < width:
        raise IOError("Can't extend image: new width %d is less, than image width %d!"
            % (new_width, width))

    x1 = new_width // 2 - width // 2
    x2 = x1 + width

    if out_channels_first:
        new_image = np.ones([channels, new_height, new_width])
        if in_channels_first:
            new_image[:, y1:y2, x1:x2] = image
        else:
            for channel in range(channels):
                new_image[channel, y1:y2, x1:x2] = image[y1:y2, x1:x2, channel]
    else:
        new_image = np.ones([new_height, new_width, channels])
        if in_channels_first:
            for channel in range(channels):
                new_image[y1:y2, x1:x2, channel] = image[channel, y1:y2, x1:x2]
        else:
            new_image[y1:y2, x1:x2, :] = image

    return new_image

def open_gray_image(path, width = None, height = None):
    """A function, that opens and processes a gray image from a path.

    Args:
        path (`string`)                 Path to an image file.
        width (`int`, optional)         Required image width        
        height (`int`, optional)        Required image height

    Returns:
        Opened image.

    """

    image = Image.open(path)

    image = ImageOps.autocontrast(image)
    image = ImageOps.grayscale(image)

    if height and width:
        size = (width, height)
    elif height:
        size = (width, image.size[1])
    elif height:
        size = (image.size[0], height)
    else:
        size = image.size

    image = ImageOps.fit(image, size, method = Image.ANTIALIAS)

    image = np.array(image)
    image = image.astype(np.float32) / 255
    image = np.expand_dims(image, axis = 0)

    return image