from resfo_utilities import CornerpointGrid, InvalidEgridFileError
import resfo
import pytest
from io import BytesIO
import numpy as np


def write_to_buffer(file_contents):
    buffer = BytesIO()
    resfo.write(buffer, file_contents)
    buffer.seek(0)
    return buffer


def test_that_read_egrid_raises_invalid_egrid_file_when_gridhead_is_mess():
    with pytest.raises(InvalidEgridFileError, match="MESS"):
        CornerpointGrid.read_egrid(write_to_buffer([("GRIDHEAD", resfo.MESS)]))


def test_that_read_egrid_raises_invalid_egrid_file_when_gridhead_is_too_short():
    with pytest.raises(InvalidEgridFileError, match="contained too few elements"):
        CornerpointGrid.read_egrid(write_to_buffer([("GRIDHEAD", [1.0])]))


def test_that_read_egrid_raises_invalid_egrid_file_when_coord_is_mess():
    with pytest.raises(InvalidEgridFileError, match="MESS"):
        CornerpointGrid.read_egrid(
            write_to_buffer([("GRIDHEAD", [1, 1, 1, 1]), ("COORD   ", resfo.MESS)])
        )


def test_that_read_egrid_raises_invalid_egrid_file_when_coord_has_too_many_values():
    with pytest.raises(InvalidEgridFileError, match="did not match grid dimensions"):
        CornerpointGrid.read_egrid(
            write_to_buffer(
                [
                    ("GRIDHEAD", [1, 1, 1, 1]),
                    ("COORD   ", [1.0]),
                    ("ZCORN   ", [1.0] * 8),
                ],
            )
        )


def test_that_read_egrid_raises_invalid_egrid_file_when_mapaxes_is_mess():
    with pytest.raises(InvalidEgridFileError, match="MESS"):
        CornerpointGrid.read_egrid(write_to_buffer([("MAPAXES ", resfo.MESS)]))


def test_that_read_egrid_raises_invalid_egrid_file_when_mapaxes_has_too_many_values():
    with pytest.raises(InvalidEgridFileError, match="contained too few elements"):
        CornerpointGrid.read_egrid(write_to_buffer([("MAPAXES ", [1.0])]))


def pad_to(lst: list[int], target_len: int):
    return np.pad(lst, (0, target_len - len(lst)), mode="constant")


@pytest.mark.parametrize(
    "contents_after_global_grid",
    [
        [],
        [
            ("LGR     ", np.array([b"LGR1    "], dtype="|S8")),
            ("LGRPARNT", np.array([b"        "], dtype="|S8")),
            (
                "GRIDHEAD",
                pad_to([1, 2, 2, 2, 1] + [0] * 19 + [1, 1, 0, 2, 2, 2, 2, 2, 2], 100),
            ),
            ("COORD   ", np.arange(200, 254)),
            ("ZCORN   ", np.arange(300, 364)),
            ("ACTNUM  ", np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=">i4")),
            ("HOSTNUM ", np.array([8, 8, 8, 8, 8, 8, 8, 8], dtype=">i4")),
            ("ENDGRID ", np.array([], dtype=">i4")),
            ("ENDLGR  ", np.array([], dtype=">i4")),
            ("NNCHEAD ", pad_to([0, 1], 10)),
            ("NNC1    ", np.array([], dtype=">i4")),
            ("NNC2    ", np.array([], dtype=">i4")),
            ("NNCL    ", np.array([], dtype=">i4")),
            ("NNCG    ", np.array([], dtype=">i4")),
        ],
    ],
)
def test_that_read_egrid_fetches_the_geometry_from_the_global_grid_in_the_file(
    contents_after_global_grid,
):
    grid = CornerpointGrid.read_egrid(
        write_to_buffer(
            [
                ("FILEHEAD", pad_to([3, 2007, 0, 0, 0, 0, 1], 100)),
                ("MAPAXES ", np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=">f4")),
                ("GRIDUNIT", np.array([b"METRES  ", b"        "], dtype="|S8")),
                ("GRIDHEAD", pad_to([1, 2, 2, 2], 100)),
                ("COORD   ", np.arange(0, 54, dtype=">f4")),
                ("ZCORN   ", np.arange(100, 164, dtype=">f4")),
                ("ACTNUM  ", np.ones((8,), dtype=">i4")),
                ("ENDGRID ", np.array([], dtype=">i4")),
                *contents_after_global_grid,
            ]
        )
    )
    assert grid.map_axes.origin == (0.0, 0.0)
    assert grid.map_axes.y_axis == (0.0, 1.0)
    assert grid.map_axes.x_axis == (1.0, 0.0)
    # coord indecies are (ni, nj, nk, 3)
    # where each triplet is the x,y,z coordinate
    # of the pillar at (i,j,k)

    # The opm manual describes the order in the input as follows:
    # "COORD defines a set of coordinate lines or pillars for a reservoir grid via
    # an array. A total of 6 x (NX+1) x (NY+1) lines must be specified for each
    # coordinate data set (or reservoir). For multiple reservoirs, where
    # NUMRES is greater than one, there must be 6 x (NX+1) x (NY+1) x NUMRES values.
    # For Cartesian geometry, each line is defined by the (x, y, z) coordinates of
    # two distinct points on the line. The lines are entered with I cycling fastest then J."
    assert grid.coord.tolist() == [
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],  # Top and base points for i,j=0,0
            [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
            [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
        ],
        [
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            [[24.0, 25.0, 26.0], [27.0, 28.0, 29.0]],
            [[42.0, 43.0, 44.0], [45.0, 46.0, 47.0]],
        ],
        [
            [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
            [[30.0, 31.0, 32.0], [33.0, 34.0, 35.0]],
            [[48.0, 49.0, 50.0], [51.0, 52.0, 53.0]],
        ],
    ]
    # coord indecies are (ni, nj, nk, 8)
    # where the 8 values describe depth of the corners
    # for cell (i,j,k)
    # Order of heights for each corner is
    # (N(orth) means higher y, E(east) means higer x, T(op) means lower z (depth))
    # TSW TSE TNW TNE BSW BSE BNW BNE

    # The opm manual describes the order of the input as follows:
    # "ZCORN defines the depth of each corner point of a grid block
    # on the pillars defining the reservoir grid. A total of 8 x NX x NY x NZ values
    # are needed to fully define all the depths in the model. The depths
    # specifying the top of the first layer are entered first with one point
    # for each pillar for each grid block. The points are entered
    # with the X axis cycling fastest. Next come the depths of the bottom
    # of the first layer. The top of layer two follows etc."
    assert grid.zcorn.tolist() == [
        [
            [
                [100.0, 101.0, 104.0, 105.0, 116.0, 117.0, 120.0, 121.0],
                [132.0, 133.0, 136.0, 137.0, 148.0, 149.0, 152.0, 153.0],
            ],
            [
                [108.0, 109.0, 112.0, 113.0, 124.0, 125.0, 128.0, 129.0],
                [140.0, 141.0, 144.0, 145.0, 156.0, 157.0, 160.0, 161.0],
            ],
        ],
        [
            [
                [102.0, 103.0, 106.0, 107.0, 118.0, 119.0, 122.0, 123.0],
                [134.0, 135.0, 138.0, 139.0, 150.0, 151.0, 154.0, 155.0],
            ],
            [
                [110.0, 111.0, 114.0, 115.0, 126.0, 127.0, 130.0, 131.0],
                [142.0, 143.0, 146.0, 147.0, 158.0, 159.0, 162.0, 163.0],
            ],
        ],
    ]


def test_that_get_xy_meshgrid_at_z_returns_meshgrid():
    coord = np.array(
        [
            [
                [[0.0, 0.0, 0.0], [1.0, 10.0, 100.0]],
                [[10.0, 20.0, 0.0], [20.0, 30.0, 100.0]],
            ]
        ]
    )
    grid = CornerpointGrid(coord, None, None)
    assert grid.get_xy_meshgrid_at_z(50.0).tolist() == [[[0.5, 5.0], [15.0, 25.0]]]


def test_that_get_xy_meshgrid_at_z_keeps_same_shape_as_coord_in_i_j_dimensions():
    coord = np.array(
        [
            [
                [[0.0, 1.0, 100.0], [0.0, 1.0, 200.0]],
                [[2.0, 3.0, 101.0], [2.0, 3.0, 201.0]],
                [[4.0, 5.0, 102.0], [4.0, 5.0, 202.0]],
            ],
            [
                [[6.0, 7.0, 103.0], [6.0, 7.0, 203.0]],
                [[8.0, 9.0, 104.0], [8.0, 9.0, 204.0]],
                [[10.0, 11.0, 105.0], [10.0, 11.0, 205.0]],
            ],
            [
                [[12.0, 13.0, 106.0], [12.0, 13.0, 206.0]],
                [[14.0, 15.0, 107.0], [14.0, 15.0, 207.0]],
                [[16.0, 17.0, 108.0], [16.0, 17.0, 208.0]],
            ],
        ]
    )
    grid = CornerpointGrid(coord, None, None)
    assert grid.get_xy_meshgrid_at_z(50.0).tolist() == [
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ],
        [
            [6.0, 7.0],
            [8.0, 9.0],
            [10.0, 11.0],
        ],
        [
            [12.0, 13.0],
            [14.0, 15.0],
            [16.0, 17.0],
        ],
    ]
