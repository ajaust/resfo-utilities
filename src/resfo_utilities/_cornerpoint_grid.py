import os
from typing import Self
from dataclasses import dataclass
from numpy import typing as npt
import numpy as np
import resfo


class InvalidEgridFileError(ValueError):
    pass


@dataclass
class CornerpointGrid:
    coord: npt.NDArray[np.float32]

    @classmethod
    def read_egrid(cls, filename: str | os.PathLike) -> Self:
        coord = None
        dims = None
        for entry in resfo.lazy_read(filename):
            match entry.read_keyword():
                case "COORD   ":
                    coord = entry.read_array()
                case "GRIDHEAD":
                    array = entry.read_array()
                    if len(array) < 4:
                        raise InvalidEgridFileError(
                            f"GRIDHEAD in EGRID file {filename} contained too few elements"
                        )
                    dims = tuple(array[1:4])

        if coord is None:
            raise InvalidEgridFileError(f"EGRID file {filename} did not contain COORD")
        if dims is None:
            raise InvalidEgridFileError(
                f"EGRID file {filename} did not contain dimensions"
            )

        return cls(np.swapaxes(coord.reshape((dims[1] + 1, dims[0] + 1, 2, 3)), 0, 1))
