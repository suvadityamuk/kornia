import keras_core as keras

from kornia.core import Module


def rgb_to_xyz(image):
    r"""Convert a RGB image to XYZ.

    .. image:: _static/img/rgb_to_xyz.png

    Args:
        image: RGB Image to be converted to XYZ with shape :math:`(*, 3, H, W)`.

    Returns:
         XYZ version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_xyz(input)  # 2x3x4x5
    """

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out = keras.ops.stack([x, y, z], axis=-3)

    return out


def xyz_to_rgb(image):
    r"""Convert a XYZ image to RGB.

    Args:
        image: XYZ Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = xyz_to_rgb(input)  # 2x3x4x5
    """

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    x = image[..., 0, :, :]
    y = image[..., 1, :, :]
    z = image[..., 2, :, :]

    r = 3.2404813432005266 * x + -1.5371515162713185 * y + -0.4985363261688878 * z
    g = -0.9692549499965682 * x + 1.8759900014898907 * y + 0.0415559265582928 * z
    b = 0.0556466391351772 * x + -0.2040413383665112 * y + 1.0573110696453443 * z

    out = keras.ops.stack([r, g, b], axis=-3)

    return out


class RgbToXyz(keras.layers.Layer): #CHECK nn.Module
    r"""Convert an image from RGB to XYZ.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        XYZ version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> xyz = RgbToXyz()
        >>> output = xyz(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    def call(self, image):
        return rgb_to_xyz(image)


class XyzToRgb(keras.layers.Layer): ##CHECK nn.Module
    r"""Converts an image from XYZ to RGB.

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = XyzToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    def call(self, image):
        return xyz_to_rgb(image)
