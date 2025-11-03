from resfo_utilities import SummaryReader, InvalidSummaryError
from resfo_utilities.testing import summaries, Unsmry
from pathlib import Path
from io import StringIO, BytesIO
import pytest
from hypothesis import given
import hypothesis.strategies as st
from contextlib import suppress
import resfo
import numpy as np


def test_that_summary_reader_can_be_initialized_with_either_path_or_io(tmp_path: Path):
    (tmp_path / "CASE.FSMSPEC").touch()
    (tmp_path / "CASE.FUNSMRY").touch()
    _ = SummaryReader(case_path=tmp_path / "CASE")
    _ = SummaryReader(smspec=StringIO, summaries=[StringIO])
    with pytest.raises(ValueError):
        _ = SummaryReader(
            case_path=tmp_path,
            smspec=StringIO,
            summaries=[StringIO],
        )
    with pytest.raises(ValueError):
        _ = SummaryReader()


@given(st.binary(), st.binary())
def test_that_summary_reader_only_raises_invalid_summary_error(
    spec: bytes, unsmry: bytes
):
    with suppress(InvalidSummaryError):
        reader = SummaryReader(
            smspec=lambda: BytesIO(spec), summaries=[lambda: BytesIO(unsmry)]
        )
        _ = list(reader.values())


def report_step_value(unsmry: Unsmry, report_step: int, kw_index: int):
    return unsmry.steps[report_step].ministeps[-1].params[kw_index]


def step_value(unsmry: Unsmry, index: int, kw_index: int):
    while index >= 0:
        for step in unsmry.steps:
            if index < len(step.ministeps):
                return step.ministeps[index].params[kw_index]
            index -= len(step.ministeps)


@given(summary=summaries(), report_step_only=st.booleans())
def test_that_the_read_values_matches_those_in_the_input(summary, report_step_only):
    smspec, unsmry = summary
    smspec_buf = BytesIO()
    unsmry_buf = BytesIO()
    smspec.to_file(smspec_buf)
    unsmry.to_file(unsmry_buf)
    smspec_buf.seek(0)
    unsmry_buf.seek(0)

    summary = SummaryReader(smspec=lambda: smspec_buf, summaries=[lambda: unsmry_buf])

    values = list(summary.values(report_step_only))
    getter = report_step_value if report_step_only else step_value
    for kw_index, _ in enumerate(summary.summary_keywords):
        for report_step, val in enumerate(values):
            assert getter(unsmry, report_step, kw_index) == pytest.approx(val[kw_index])


def read_summary(smspec, unsmry, report_step_only=True):
    summary = SummaryReader(smspec=lambda: smspec, summaries=[lambda: unsmry])
    _ = list(summary.values(report_step_only))
    _ = list(summary.summary_keywords)


@pytest.mark.parametrize(
    "spec_contents, smry_contents, error_message",
    [
        (b"", b"", "Keyword startdat missing"),
        (b"1", b"1", "Summary files contained invalid contents"),
        (
            b"\x00\x00\x00\x10FOOOOOOO\x00\x00\x00\x01"
            + b"INTE"
            + b"\x00\x00\x00\x10"
            + (4).to_bytes(4, signed=True, byteorder="big")
            + b"\x00" * 4,
            b"",
            "Keyword startdat missing",
        ),
        (
            b"\x00\x00\x00\x10STARTDAT\x00\x00\x00\x01"
            + b"INTE"
            + b"\x00\x00\x00\x10"
            + (4).to_bytes(4, signed=True, byteorder="big")
            + b"\x00" * 4
            + (4).to_bytes(4, signed=True, byteorder="big"),
            b"",
            "contains invalid STARTDAT",
        ),
    ],
)
def test_that_incorrect_summary_files_raises_informative_errors(
    smry_contents, spec_contents, error_message
):
    smry_buf = BytesIO(smry_contents)
    smry_buf.seek(0)
    spec_buf = BytesIO(spec_contents)
    spec_buf.seek(0)

    with pytest.raises(InvalidSummaryError, match=error_message):
        read_summary(spec_buf, smry_buf)


@given(summaries())
def test_truncated_summary_file_raises_invalidresponsefile(summary):
    smspec, unsmry = summary
    smspec_buf = BytesIO()
    unsmry_buf = BytesIO()
    smspec.to_file(smspec_buf)
    unsmry.to_file(unsmry_buf)
    smspec_buf.seek(0)
    unsmry_buf.seek(0)
    unsmry_buf.truncate(10)

    with pytest.raises(InvalidSummaryError, match=""):
        read_summary(smspec_buf, unsmry_buf)


def test_mess_values_in_summary_files_raises_informative_errors():
    smspec_buf = BytesIO()
    resfo.write(smspec_buf, [("KEYWORDS", resfo.MESS)])
    smspec_buf.seek(0)

    with pytest.raises(InvalidSummaryError, match="has incorrect type MESS"):
        read_summary(smspec_buf, BytesIO())


def test_missing_keywords_in_smspec_raises_informative_error():
    smspec_buf = BytesIO()
    resfo.write(
        smspec_buf,
        [
            ("STARTDAT", np.array([31, 12, 2012, 00], dtype=np.int32)),
            ("UNITS   ", ["ANNUAL  "]),
        ],
    )
    smspec_buf.seek(0)

    with pytest.raises(
        InvalidSummaryError,
        match="Keywords missing",
    ):
        read_summary(smspec_buf, BytesIO())
