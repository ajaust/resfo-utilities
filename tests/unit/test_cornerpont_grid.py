from resfo_utilities import CornerpointGrid, InvalidEgridFileError
import resfo
import pytest
from io import BytesIO


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
                [("GRIDHEAD", [1, 1, 1, 1]), ("COORD   ", [1.0])],
            )
        )
