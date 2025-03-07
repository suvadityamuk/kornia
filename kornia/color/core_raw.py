from enum import Enum

import keras_core as keras

class CFA(Enum):
    r"""Define the configuration of the color filter array.

    So far only bayer images is supported and the enum sets the pixel order for bayer. Note that this can change due
    to things like rotations and cropping of images. Take care if including the translations in pipeline.
    This implementations is optimized to be reasonably fast, look better than simple nearest neighbour.
    On top of this care is taken to make it reversible going raw -> rgb -> raw. the raw samples remain intact
    during conversion and only unknown samples are interpolated.

    The names are based on the OpenCV convention where the BG indicates pixel 1,1 (counting from 0,0) is
    blue and its neighbour to the right is green. In that case the top left pixel is red. Other options are GB, RG and
    GR

    reference:
        https://en.wikipedia.org/wiki/Color_filter_array
    """

    BG = 0
    GB = 1
    RG = 2
    GR = 3


def raw_to_rgb(image, cfa: CFA):
    r"""Convert a raw bayer image to RGB version of image.

    We are assuming a CFA with 2 green, 1 red, 1 blue. A bilinear interpolation is used for R/G and a fix convolution
    for the green pixels. To simplify calculations we expect the Height Width to be evenly divisible by 2.

    The image data is assumed to be in the range of (0, 1). Image H/W is assumed to be evenly divisible by 2.
    for simplicity reasons

    Args:
        image: raw image to be converted to RGB with shape :math:`(*,1,H,W)`.
        cfa: The configuration of the color filter.
    Returns:
        RGB version of the image with shape :math:`(*,3,H,W)`.

    Example:
        >>> rawinput = torch.randn(2, 1, 4, 6)
        >>> rgb = raw_to_rgb(rawinput, CFA.RG) # 2x3x4x6
    """

    if keras.ops.ndim(image) < 3 or keras.ops.shape(image)[-3] != 1:
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). Got {image.shape}.")

    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {image.shape}")

    imagesize = keras.ops.shape(image)

    image = image.view(-1, 1, image.shape[-2], image.shape[-1])

    # BG is defined as pel 1,1 being blue, that is the top left is actually green. This matches
    # opencv naming so makes sense to keep
    if cfa == CFA.BG:
        r = image[..., :, ::2, ::2]
        b = image[..., :, 1::2, 1::2]
        rpad = (0, 1, 0, 1)
        bpad = (1, 0, 1, 0)
    elif cfa == CFA.GB:
        r = image[..., :, ::2, 1::2]
        b = image[..., :, 1::2, ::2]
        rpad = (1, 0, 0, 1)
        bpad = (0, 1, 1, 0)
    elif cfa == CFA.RG:
        r = image[..., :, 1::2, 1::2]
        b = image[..., :, ::2, ::2]
        rpad = (1, 0, 1, 0)
        bpad = (0, 1, 0, 1)
    elif cfa == CFA.GR:
        r = image[..., :, 1::2, ::2]
        b = image[..., :, ::2, 1::2]
        rpad = (0, 1, 1, 0)
        bpad = (1, 0, 0, 1)
    else:
        raise ValueError(f"Unsupported CFA Got {cfa}.")

    # upscaling r and b with bi-linear gives reasonable quality
    # Note that depending on where these are sampled we need to pad appropriately
    # the bilinear filter will pretty much be based on for example this layout (RG)
    # (which needs to be padded bottom right)
    # +-+-+
    # |B| |
    # | | |
    # +-+-+
    # While in this layout we need to pad with additional B samples top left to
    # make sure we interpolate from the correct position
    # +-+-+
    # | | |
    # | |B|
    # +-+-+
    # For an image like this (3x2 blue pixels)
    # +------+
    # |B B B |
    # |      |
    # |B B B |
    # |      |
    # +------+
    # It needs to be expanded to this (4x3 pixels scaled to 7x5 for correct interpolation)
    # +-------+
    # |B B B b|
    # |       |
    # |B B B b|
    # |       |
    # |b b b b|
    # +-------+
    # and we crop the area afterwards. This is since the interpolation will be between first and last pixel
    # evenly spaced between them while the B/R samples will be missing in the corners were they are assumed to exist
    # Further we need to do align_corners to start the interpolation from the middle of the samples in the corners, that
    # way we get to keep the known blue samples across the whole image
    rpadded = keras.ops.pad(r, list(rpad), mode="symmetric")
    bpadded = keras.ops.pad(b, list(bpad), mode="symmetric")
    # use explicit padding instead of conv2d padding to be able to use reflect which mirror the correct colors
    # for a 2x2 bayer filter
    gpadded = keras.ops.pad(image, [1, 1, 1, 1], "reflect")

    r_up = keras.ops.image.resize(
        rpadded, (image.shape[-2] + 1, image.shape[-1] + 1), mode='bilinear', 
    ) #CHECK - align_corners=True not there in keras core
    b_up = keras.ops.image.resize(
        bpadded, (image.shape[-2] + 1, image.shape[-1] + 1), mode='bilinear', 
    ) #CHECK - align_corners=True not there in keras core

    # remove the extra padding
    r_up = keras.ops.pad(r_up, [-x for x in rpad])
    b_up = keras.ops.pad(b_up, [-x for x in bpad])

    # all unknown pixels are the average of the nearby green samples
    kernel = keras.ops.convert_to_tensor(
        [[[[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]]]], dtype=image.dtype
    )

    # This is done on all samples but result for the known green samples is then overwritten by the input
    g_up = keras.ops.conv(gpadded, kernel)

    # overwrite the already known samples which otherwise have values from r/b
    # this depends on the CFA configuration
    if cfa == CFA.BG:
        g_up[:, :, ::2, 1::2] = image[:, :, ::2, 1::2]
        g_up[:, :, 1::2, ::2] = image[:, :, 1::2, ::2]
    elif cfa == CFA.GB:
        g_up[:, :, ::2, ::2] = image[:, :, ::2, ::2]
        g_up[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    elif cfa == CFA.RG:
        g_up[:, :, 1::2, ::2] = image[:, :, 1::2, ::2]
        g_up[:, :, ::2, 1::2] = image[:, :, ::2, 1::2]
    elif cfa == CFA.GR:
        g_up[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
        g_up[:, :, ::2, ::2] = image[:, :, ::2, ::2]
    else:
        raise ValueError(f"Unsupported CFA Got {cfa}.")

    r_up = r_up.view(imagesize)
    g_up = g_up.view(imagesize)
    b_up = b_up.view(imagesize)

    rgb = keras.ops.concatenate([r_up, g_up, b_up], axis=-3)
    return rgb


def rgb_to_raw(image, cfa: CFA):
    r"""Convert a RGB image to RAW version of image with the specified color filter array.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB image to be converted to bayer raw with shape :math:`(*,3,H,W)`.
        cfa: Which color filter array do we want the output to mimic. I.e. which pixels are red/green/blue.

    Returns:
        raw version of the image with shape :math:`(*,1,H,W)`.

    Example:
        >>> rgbinput = torch.rand(2, 3, 4, 6)
        >>> raw = rgb_to_raw(rgbinput, CFA.BG) # 2x1x4x6
    """

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # pick the tensor with green pixels
    # clone to make sure grad works
    output = image[..., 1:2, :, :].clone()

    # overwrite the r/b positions (depending on the cfa configuration) with blue/red pixels
    if cfa == CFA.BG:
        output[..., :, ::2, ::2] = image[..., 0:1, ::2, ::2]  # red
        output[..., :, 1::2, 1::2] = image[..., 2:3, 1::2, 1::2]  # blue
    elif cfa == CFA.GB:
        output[..., :, ::2, 1::2] = image[..., 0:1, ::2, 1::2]  # red
        output[..., :, 1::2, ::2] = image[..., 2:3, 1::2, ::2]  # blue
    elif cfa == CFA.RG:
        output[..., :, 1::2, 1::2] = image[..., 0:1, 1::2, 1::2]  # red
        output[..., :, ::2, ::2] = image[..., 2:3, ::2, ::2]  # blue
    elif cfa == CFA.GR:
        output[..., :, 1::2, ::2] = image[..., 0:1, 1::2, ::2]  # red
        output[..., :, ::2, 1::2] = image[..., 2:3, ::2, 1::2]  # blue

    return output


class RawToRgb(keras.layers.Layer):  #CHECK nn.module
    r"""Module to convert a bayer raw image to RGB version of image.

    The image data is assumed to be in the range of (0, 1).

    Shape:
        - image: :math:`(*, 1, H, W)`
        - output: :math:`(*, 3, H, W)`

    Example:
        >>> rawinput = torch.rand(2, 1, 4, 6)
        >>> rgb = RawToRgb(CFA.RG)
        >>> output = rgb(rawinput)  # 2x3x4x5
    """

    def __init__(self, cfa: CFA) -> None:
        super().__init__()
        self.cfa = cfa

    def call(self, image):
        return raw_to_rgb(image, cfa=self.cfa)


class RgbToRaw(keras.layers.Layer): #CHECK nn.module
    r"""Module to convert a RGB image to bayer raw version of image.

    The image data is assumed to be in the range of (0, 1).

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)`

    reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Example:
        >>> rgbinput = torch.rand(2, 3, 4, 6)
        >>> raw = RgbToRaw(CFA.GB)
        >>> output = raw(rgbinput)  # 2x1x4x6
    """

    def __init__(self, cfa: CFA) -> None:
        super().__init__()
        self.cfa = cfa

    def call(self, image):
        return rgb_to_raw(image, cfa=self.cfa)
