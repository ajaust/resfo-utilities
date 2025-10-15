from resfo_utilities._geometry import point_in_hexahedron
from hypothesis import given
import hypothesis.strategies as st

unit_cube = [
    [0.0] * 3,
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
]
unit_quads = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 4, 5, 1],
    [1, 5, 6, 2],
    [2, 6, 7, 3],
    [3, 7, 4, 0],
]


unit_coordinates = st.floats(min_value=0.0, max_value=1.0)


@given(st.tuples(*([unit_coordinates] * 3)))
def test_that_unit_interval_coordinates_are_inside_the_unit_cube(point):
    assert point_in_hexahedron(unit_cube, point, unit_quads)


def test_that_cell_beyond_hexahedron_is_not_inside():
    assert point_in_hexahedron(unit_cube, [1.5] * 3, unit_quads) is False
    assert point_in_hexahedron(unit_cube, [-1.5] * 3, unit_quads) is False
    for j in [1, -1]:
        for i in range(3):
            point = [0.5, 0.5, 0.5]
            point[i] = j * 1.5
            assert point_in_hexahedron(unit_cube, point, unit_quads) is False


def test_that_quad_can_be_given_in_cw_order():
    cw_unit_quads = [
        list(reversed([0, 1, 2, 3])),
        list(reversed([4, 5, 6, 7])),
        list(reversed([0, 4, 5, 1])),
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ]

    assert point_in_hexahedron(unit_cube, [0.5] * 3, cw_unit_quads)
    assert point_in_hexahedron(unit_cube, [1.5] * 3, cw_unit_quads) is False
    assert point_in_hexahedron(unit_cube, [-1.5] * 3, cw_unit_quads) is False


def test_that_points_on_the_boundary_are_considered_inside_the_hexahedron():
    assert point_in_hexahedron(unit_cube, [1.0] * 3, unit_quads)
