import pytest
import shutil


@pytest.fixture
def simulator_cmd():
    simulator_cmd = shutil.which("flow")
    if not simulator_cmd:
        pytest.skip(reason="flow not available")
    return [simulator_cmd]
