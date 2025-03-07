"""The testing package contains testing-specific utilities."""
from __future__ import annotations

from typing import Any, Sequence, TypeVar, cast

from torch import float16, float32, float64
from typing_extensions import TypeGuard

__all__ = [
    "KORNIA_CHECK_SHAPE",
    "KORNIA_CHECK",
    "KORNIA_UNWRAP",
    "KORNIA_CHECK_TYPE",
    "KORNIA_CHECK_IS_TENSOR",
    "KORNIA_CHECK_IS_LIST_OF_TENSOR",
    "KORNIA_CHECK_SAME_DEVICE",
    "KORNIA_CHECK_SAME_DEVICES",
    "KORNIA_CHECK_IS_COLOR",
    "KORNIA_CHECK_IS_GRAY",
    "KORNIA_CHECK_IS_IMAGE",
    "KORNIA_CHECK_DM_DESC",
    "KORNIA_CHECK_LAF",
]

# Logger api


# TODO: add somehow type check, or enforce to do it before
def KORNIA_CHECK_SHAPE(x, shape: list[str], raises: bool = True) -> bool:
    """Check whether a tensor has a specified shape.

    The shape can be specified with a implicit or explicit list of strings.
    The guard also check whether the variable is a type `IntegratedTensor`.

    Args:
        x: the tensor to evaluate.
        shape: a list with strings with the expected shape.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        Exception: if the input tensor is has not the expected shape and raises is True.

    Example:
        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["B", "C", "H", "W"])  # implicit
        True

        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["2", "3", "H", "W"])  # explicit
        True
    """
    if '*' == shape[0]:
        shape_to_check = shape[1:]
        x_shape_to_check = x.shape[-len(shape) + 1 :]
    elif '*' == shape[-1]:
        shape_to_check = shape[:-1]
        x_shape_to_check = x.shape[: len(shape) - 1]
    else:
        shape_to_check = shape
        x_shape_to_check = x.shape

    if len(x_shape_to_check) != len(shape_to_check):
        if raises:
            raise TypeError(f"{x} shape must be [{shape}]. Got {x.shape}")
        else:
            return False

    for i in range(len(x_shape_to_check)):
        # The voodoo below is because torchscript does not like
        # that dim can be both int and str
        dim_: str = shape_to_check[i]
        if not dim_.isnumeric():
            continue
        dim = int(dim_)
        if x_shape_to_check[i] != dim:
            if raises:
                raise TypeError(f"{x} shape must be [{shape}]. Got {x.shape}")
            else:
                return False
    return True


def KORNIA_CHECK(condition: bool, msg: str | None = None, raises: bool = True) -> bool:
    """Check any arbitrary boolean condition.

    Args:
        condition: the condition to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        Exception: if the condition is met and raises is True.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK(x.shape[-2:] == (3, 3), "Invalid homography")
        True
    """
    if not condition:
        if raises:
            raise Exception(f"{condition} not true.\n{msg}")
        return False
    return True


def KORNIA_UNWRAP(maybe_obj: object, typ: Any) -> Any:
    """Unwraps an optional contained value that may or not be present.

    Args:
        maybe_obj: the object to unwrap.
        typ: expected type after unwrap.
    """
    # TODO: this function will change after kornia/pr#1987
    return cast(typ, maybe_obj)


T = TypeVar('T', bound=type)


# TODO: fix mypy typeguard issue
def KORNIA_CHECK_TYPE(x: object, typ: T | tuple[T, ...], msg: str | None = None, raises: bool = True) -> TypeGuard[T]:
    """Check the type of an aribratry variable.

    Args:
        x: any input variable.
        typ: the expected type of the variable.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeException: if the input variable does not match with the expected and raises is True.

    Example:
        >>> KORNIA_CHECK_TYPE("foo", str, "Invalid string")
        True
    """
    # TODO: Move to use typeguard here dropping support for JIT
    if not isinstance(x, typ):
        if raises:
            raise TypeError(f"Invalid type: {type(x)}.\n{msg}")
        return False
    return True


#TODO: HOW TO HANDLE?
def KORNIA_CHECK_IS_TENSOR(x: object, msg: str | None = None, raises: bool = True):
    """Check the input variable is a Tensor.

    Args:
        x: any input variable.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeException: if the input variable does not match with the expected and raises is True.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_IS_TENSOR(x, "Invalid tensor")
        True
    """
    # TODO: Move to use typeguard here dropping support for JIT
    # if not isinstance(x, IntegratedTensor):
    #     if raises:
    #         raise TypeError(f"Not a IntegratedTensor type. Got: {type(x)}.\n{msg}")
    #     return False
    return True

#TODO: HOW TO HANDLE?
def KORNIA_CHECK_IS_LIST_OF_TENSOR(x: Sequence[object] | None, raises: bool = True):
    """Check the input variable is a List of IntegratedTensors.

    Args:
        x: Any sequence of objects
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeException: if the input variable does not match with the expected and raises is True.

    Return:
        True if the input is a list of IntegratedTensors, otherwise return False.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_IS_LIST_OF_TENSOR(x, raises=False)
        False
        >>> KORNIA_CHECK_IS_LIST_OF_TENSOR([x])
        True
    """
    # are_tensors = isinstance(x, list) and all(isinstance(d, IntegratedTensor) for d in x)
    # if not are_tensors:
    #     if raises:
    #         raise TypeError(f"Provided container of type {type(x)} is not a list of tensors")
    #     return False
    return True


def KORNIA_CHECK_SAME_DEVICE(x, y, raises: bool = True) -> bool:
    """Check whether two tensor in the same device.

    Args:
        x: first tensor to evaluate.
        y: sencod tensor to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeException: if the two tensors are not in the same device and raises is True.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(1, 3, 1)
        >>> KORNIA_CHECK_SAME_DEVICE(x1, x2)
        True
    """
    # if x.device != y.device:
    #     if raises:
    #         raise TypeError(f"Not same device for tensors. Got: {x.device} and {y.device}")
    #     return False
    return True


def KORNIA_CHECK_SAME_DEVICES(tensors, msg: str | None = None, raises: bool = True) -> bool:
    """Check whether a list provided tensors live in the same device.

    Args:
        x: a list of tensors.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        Exception: if all the tensors are not in the same device and raises is True.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(1, 3, 1)
        >>> KORNIA_CHECK_SAME_DEVICES([x1, x2], "IntegratedTensors not in the same device")
        True
    """
    KORNIA_CHECK(isinstance(tensors, list) and len(tensors) >= 1, "Expected a list with at least one element", raises)
    if not all(tensors[0].device == x.device for x in tensors):
        if raises:
            raise Exception(f"Not same device for tensors. Got: {[x.device for x in tensors]}.\n{msg}")
        return False
    return True


def KORNIA_CHECK_SAME_SHAPE(x, y, raises: bool = True) -> bool:
    """Check whether two tensor have the same shape.

    Args:
        x: first tensor to evaluate.
        y: sencod tensor to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeException: if the two tensors have not the same shape and raises is True.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_SAME_SHAPE(x1, x2)
        True
    """
    if x.shape != y.shape:
        if raises:
            raise TypeError(f"Not same shape for tensors. Got: {x.shape} and {y.shape}")
        return False
    return True


def KORNIA_CHECK_IS_COLOR(x, msg: str | None = None, raises: bool = True) -> bool:
    """Check whether an image tensor is a color images.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeException: if all the input tensor has not a shape :math:`(3,H,W)` and raises is True.

    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_IS_COLOR(img, "Image is not color")
        True
    """
    if len(x.shape) < 3 or x.shape[-3] != 3:
        if raises:
            raise TypeError(f"Not a color tensor. Got: {type(x)}.\n{msg}")
        return False
    return True


def KORNIA_CHECK_IS_GRAY(x, msg: str | None = None, raises: bool = True) -> bool:
    """Check whether an image tensor is grayscale.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeException: if the tensor has not a shape :math:`(1,H,W)` or :math:`(H,W)` and raises is True.

    Example:
        >>> img = torch.rand(2, 1, 4, 4)
        >>> KORNIA_CHECK_IS_GRAY(img, "Image is not grayscale")
        True
    """
    if len(x.shape) < 2 or (len(x.shape) >= 3 and x.shape[-3] != 1):
        if raises:
            raise TypeError(f"Not a gray tensor. Got: {type(x)}.\n{msg}")
        return False
    return True


def KORNIA_CHECK_IS_COLOR_OR_GRAY(x, msg: str | None = None, raises: bool = True) -> bool:
    """Check whether an image tensor is grayscale or color.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeException: if the tensor has not a shape :math:`(1,H,W)` or :math:`(3,H,W)` and raises is True.

    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_IS_COLOR_OR_GRAY(img, "Image is not color orgrayscale")
        True
    """
    if len(x.shape) < 3 or x.shape[-3] not in [1, 3]:
        if raises:
            raise TypeError(f"Not a color or gray tensor. Got: {type(x)}.\n{msg}")
        return False
    return True


def KORNIA_CHECK_IS_IMAGE(x, msg: str | None = None, raises: bool = True, bits: int = 8) -> bool:
    """Check whether an image tensor is ranged properly [0, 1] for float or [0, 2 ** bits] for int.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.
        bits: the image bits. The default checks if given integer input image is an
            8-bit image (0-255) or not.

    Raises:
        TypeException: if all the input tensor has not 1) a shape :math:`(3,H,W)`,
        2) [0, 1] for float or [0, 255] for int, 3) and raises is True.

    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_IS_IMAGE(img, "It is not an image")
        True
    """
    res = KORNIA_CHECK_IS_COLOR_OR_GRAY(x, msg, raises=raises)

    if not raises and not res:
        return False

    err_msg = f"Invalid image value range. Expect [0, 1] but got [{x.min()}, {x.max()}]."
    if msg is not None:
        err_msg += f"\n{msg}"

    if x.dtype in [float16, float32, float64] and (x.min() < 0.0 or x.max() > 1.0):
        if raises:
            raise ValueError(err_msg)
        return False
    elif x.min() < 0 or x.max() > 2**bits - 1:
        if raises:
            raise ValueError(err_msg)
        return False
    return True


def KORNIA_CHECK_DM_DESC(desc1, desc2, dm, raises: bool = True) -> bool:
    """Check whether the provided descriptors match with a distance matrix.

    Args:
        desc1: first descriptor tensor to evaluate.
        desc2: second descriptor tensor to evaluate.
        dm: distance matrix tensor to evaluate.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeException: if the descriptors shape do not match with the distance matrix and raises is True.

    Example:
        >>> desc1 = torch.rand(4)
        >>> desc2 = torch.rand(8)
        >>> dm = torch.rand(4, 8)
        >>> KORNIA_CHECK_DM_DESC(desc1, desc2, dm)
        True
    """
    if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
        if raises:
            raise TypeError(
                f"distance matrix shape {dm.shape} is not onsistent with descriptors shape: desc1 {desc1.shape} "
                f"desc2 {desc2.shape}"
            )
        return False
    return True


def KORNIA_CHECK_LAF(laf, raises: bool = True) -> bool:
    """Check whether a Local Affine Frame (laf) has a valid shape.

    Args:
        laf: local affine frame tensor to evaluate.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        Exception: if the input laf does not have a shape :math:`(B,N,2,3)` and raises is True.

    Example:
        >>> lafs = torch.rand(2, 10, 2, 3)
        >>> KORNIA_CHECK_LAF(lafs)
        True
    """
    return KORNIA_CHECK_SHAPE(laf, ["B", "N", "2", "3"], raises)
