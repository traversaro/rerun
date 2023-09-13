# DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/python.rs
# Based on "crates/re_types/definitions/rerun/components/instance_key.fbs".


from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attrs import define, field

from .._baseclasses import (
    BaseExtensionArray,
    BaseExtensionType,
)
from ._overrides import instance_key__native_to_pa_array_override  # noqa: F401

__all__ = ["InstanceKey", "InstanceKeyArray", "InstanceKeyArrayLike", "InstanceKeyLike", "InstanceKeyType"]


@define
class InstanceKey:
    """A unique numeric identifier for each individual instance within a batch."""

    # You can define your own __init__ function by defining a function called "instance_key__init_override"

    value: int = field(converter=int)

    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        # You can replace `np.asarray` here with your own code by defining a function named "instance_key__as_array_override"
        return np.asarray(self.value, dtype=dtype)

    def __int__(self) -> int:
        return int(self.value)


if TYPE_CHECKING:
    InstanceKeyLike = Union[InstanceKey, int]
else:
    InstanceKeyLike = Any

InstanceKeyArrayLike = Union[InstanceKey, Sequence[InstanceKeyLike], int, npt.NDArray[np.uint64]]


# --- Arrow support ---


class InstanceKeyType(BaseExtensionType):
    def __init__(self) -> None:
        pa.ExtensionType.__init__(self, pa.uint64(), "rerun.components.InstanceKey")


class InstanceKeyArray(BaseExtensionArray[InstanceKeyArrayLike]):
    _EXTENSION_NAME = "rerun.components.InstanceKey"
    _EXTENSION_TYPE = InstanceKeyType

    @staticmethod
    def _native_to_pa_array(data: InstanceKeyArrayLike, data_type: pa.DataType) -> pa.Array:
        return instance_key__native_to_pa_array_override(data, data_type)


InstanceKeyType._ARRAY_TYPE = InstanceKeyArray

# TODO(cmc): bring back registration to pyarrow once legacy types are gone
# pa.register_extension_type(InstanceKeyType())
