import pytest
import shutil
import os


@pytest.fixture
def simulator_cmd():
    simulator_cmd = os.environ.get("TEST_SIMULATOR_CMD") or shutil.which("flow")
    if not simulator_cmd:
        pytest.skip(
            reason="Did not find OPM flow simulator"
            " and no simulator set in TEST_SIMULATOR_CMD"
        )
    return simulator_cmd.split()
