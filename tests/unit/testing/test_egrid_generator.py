from resfo_utilities.testing import GlobalGrid
from hypothesis import given
import hypothesis.strategies as st


@given(st.builds(GlobalGrid))
def test_that_global_grid_eq_is_reflexive(global_grid):
    assert global_grid == global_grid


@given(*([st.builds(GlobalGrid)] * 3))
def test_that_global_grid_eq_is_transitive(a, b, c):
    if a == b and b == c:
        assert a == c
