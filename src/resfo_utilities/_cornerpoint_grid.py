import os
from typing import Self, Any, IO, TypeVar
from dataclasses import dataclass
from numpy import typing as npt
import numpy as np
import resfo
import trimesh
from itertools import product
from matplotlib.path import Path


class InvalidEgridFileError(ValueError):
    pass


@dataclass
class MapAxes:
    y_axis: tuple[np.float32, np.float32]
    origin: tuple[np.float32, np.float32]
    x_axis: tuple[np.float32, np.float32]

    def transform_map_points(
        self, points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Transforms points from map coordinates to grid coordinates."""
        translated = points - np.array([*self.origin, 0])
        tx = translated[:, 0]
        ty = translated[:, 1]
        x_vec = (self.x_axis[0] - self.origin[0], self.x_axis[1] - self.origin[1])
        y_vec = (self.y_axis[0] - self.origin[0], self.y_axis[1] - self.origin[1])
        x_norm = np.sqrt(x_vec[0] ** 2 + x_vec[1] ** 2)
        x_unit = (x_vec[0] / x_norm, x_vec[1] / x_norm)
        y_norm = np.sqrt(y_vec[0] ** 2 + y_vec[1] ** 2)
        y_unit = (y_vec[0] / y_norm, y_vec[1] / y_norm)
        norm = 1.0 / (x_unit[0] * y_unit[1] - x_unit[1] * y_unit[0])
        return np.column_stack(
            [
                (tx * y_unit[1] - ty * y_unit[0]) * norm,
                (-tx * x_unit[1] + ty * x_unit[0]) * norm,
                translated[:, 2],
            ]
        )


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

    def find_cell_containing_point(
        self, points: npt.ArrayLike, map_coordinates: bool = True
    ) -> list[tuple[float, float, float] | None]:
        points = np.asarray(points)
        result: list[tuple[float, float, float] | None] = []
        if map_coordinates and self.map_axes is not None:
            points = self.map_axes.transform_map_points(points)
        found = False
        for p in points:
            mesh = self._pillars_z_plane_intersection(p[2])
            for i, j in product(*map(range, self.zcorn.shape[0:2])):
                if Path(
                    [
                        mesh[i, j],
                        mesh[i + 1, j],
                        mesh[i + 1, j + 1],
                        mesh[i, j + 1],
                    ]
                ).contains_points([p[0:2]]):
                    for k in range(self.zcorn.shape[2]):
                        zcorn = self.zcorn[i, j, k]
                        max_z, min_z = zcorn.max(), zcorn.min()
                        if min_z <= p[2] <= max_z and self.point_in_cell(
                            p, i, j, k, map_coordinates=False
                        ):
                            result.append((i, j, k))
                            found = True
                            break
                    break
            if not found:
                result.append(None)

        return result

    def point_in_cell(
        self,
        points: npt.ArrayLike,
        i: int,
        j: int,
        k: int,
        tolerance: float = 1e-6,
        map_coordinates: bool = True,
    ) -> npt.NDArray[np.bool_]:
        """Whether the points (x,y,z) is in the cell at (i,j,k).

        Param:
            points: x,y,z triple or array of x,y,z triples to be tested for containment.
            tolerance: The minimum distance for points near the boundary to decide
                containment.
            map_coordinates: Whether the given points are in the mapaxes coordinate system,
                defaults to true.

        Returns:
            Array of boolean values for each triplet describing whether
            it is contained in the cell.
        """
        points = np.asarray(points)
        if len(points.shape) == 1:
            points = points[np.newaxis, :]
        if map_coordinates and self.map_axes is not None:
            points = self.map_axes.transform_map_points(points)
        pillar_vertices = np.concatenate(
            [
                self.coord[i, j, :],
                self.coord[i, j + 1, :],
                self.coord[i + 1, j, :],
                self.coord[i + 1, j + 1, :],
            ]
        )
        top = pillar_vertices[::2]
        bot = pillar_vertices[1::2]
        top_z = top[:, 2]
        bot_z = bot[:, 2]

        def twice(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
            return np.concatenate([a, a])

        t = (self.zcorn[i, j, k] - twice(top_z)) / twice(bot_z - top_z)
        mesh = trimesh.Trimesh(
            vertices=twice(top) + t[:, np.newaxis] * twice(bot - top),
            faces=np.array(
                [
                    [0, 1, 2],
                    [1, 2, 3],
                    [0, 1, 5],
                    [0, 4, 5],
                    [0, 2, 6],
                    [0, 4, 6],
                    [4, 6, 5],
                    [5, 6, 7],
                    [1, 3, 7],
                    [1, 5, 7],
                    [2, 3, 7],
                    [2, 3, 6],
                ]
            ),
        )
        return mesh.contains(points) | (mesh.nearest.on_surface(points)[1] < tolerance)

    def _pillars_z_plane_intersection(self, z: np.float32) -> npt.NDArray[np.float32]:
        shape = self.coord.shape
        coord = self.coord.reshape(shape[0] * shape[1], shape[2] * shape[3])
        x1, y1, z1, x2, y2, z2 = coord.T
        t = (z - z1) / (z2 - z1)

        # Compute x and y for all lines
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        # Result: (x, y) coordinates for all lines at z
        result = np.column_stack((x, y))
        return result.reshape(shape[0], shape[1], 2)
