import os
from typing import Self, Any, IO, TypeVar
from dataclasses import dataclass
from numpy import typing as npt
import numpy as np
import resfo


class InvalidEgridFileError(ValueError):
    pass


@dataclass
class MapAxes:
    y_axis: tuple[np.float32, np.float32]
    origin: tuple[np.float32, np.float32]
    x_axis: tuple[np.float32, np.float32]


@dataclass
class CornerpointGrid:
    coord: npt.NDArray[np.float32]
    zcorn: npt.NDArray[np.float32]
    map_axes: MapAxes | None = None

    @classmethod
    def read_egrid(cls, file_like: str | os.PathLike[str] | IO[Any]) -> Self:
        coord = None
        dims = None
        zcorn = None
        opened = False
        stream = None
        map_axes = None

        try:
            if isinstance(file_like, str):
                filename = file_like
                mode = "rt" if filename.lower().endswith("fegrid") else "rb"
                stream = open(filename, mode=mode)
                opened = True
            elif isinstance(file_like, os.PathLike):
                filename = str(file_like)
                mode = "rt" if filename.lower().endswith("fegrid") else "rb"
                stream = open(filename, mode=mode)
                opened = True
            else:
                filename = getattr(file_like, "name", "unknown stream")
                stream = file_like

            T = TypeVar("T", bound=np.generic)

            def validate_array(
                name: str,
                array: npt.NDArray[T] | resfo.MESS,
                min_length: int | None = None,
            ) -> npt.NDArray[T]:
                if array is resfo.MESS or isinstance(array, resfo.MESS):
                    raise InvalidEgridFileError(
                        f"Expected Array for keyword {name} in {filename} but got MESS"
                    )
                if min_length is not None and len(array) < min_length:
                    raise InvalidEgridFileError(
                        f"{name} in EGRID file {filename} contained too few elements"
                    )

                return array

            for entry in resfo.lazy_read(stream):
                kw = entry.read_keyword()
                match kw:
                    case "ZCORN   ":
                        zcorn = validate_array(kw, entry.read_array())
                    case "COORD   ":
                        coord = validate_array(kw, entry.read_array())
                    case "GRIDHEAD":
                        array = validate_array(kw, entry.read_array(), 4)
                        dims = tuple(array[1:4])
                    case "MAPAXES ":
                        array = validate_array(kw, entry.read_array(), 6)
                        map_axes = MapAxes(
                            (array[0], array[1]),
                            (array[2], array[3]),
                            (array[4], array[5]),
                        )
                    case "ENDGRID ":
                        break

            if coord is None:
                raise InvalidEgridFileError(
                    f"EGRID file {filename} did not contain COORD"
                )
            if zcorn is None:
                raise InvalidEgridFileError(
                    f"EGRID file {filename} did not contain ZCORN"
                )
            if dims is None:
                raise InvalidEgridFileError(
                    f"EGRID file {filename} did not contain dimensions"
                )
        except resfo.ResfoParsingError as err:
            raise InvalidEgridFileError(f"Could not parse EGRID file: {err}") from err
        finally:
            if opened and stream is not None:
                stream.close()
        try:
            coord = np.swapaxes(coord.reshape((dims[1] + 1, dims[0] + 1, 2, 3)), 0, 1)
        except ValueError as err:
            raise InvalidEgridFileError(
                f"COORD size {len(coord)} did not match"
                f" grid dimensions {dims} in {filename}"
            ) from err
        try:
            zcorn = zcorn.reshape(2, dims[0], 2, dims[1], 2, dims[2], order="F")
            zcorn = np.moveaxis(zcorn, [1, 3, 5, 4, 2], [0, 1, 2, 3, 4])
            zcorn = zcorn.reshape((dims[0], dims[1], dims[2], 8))
        except ValueError as err:
            raise InvalidEgridFileError(
                f"ZCORN size {len(zcorn)} did not match"
                f" grid dimensions {dims} in {filename}"
            ) from err
        return cls(coord, zcorn, map_axes)
