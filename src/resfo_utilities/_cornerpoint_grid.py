import os
from typing import Self


class CornerpointGrid:
    @classmethod
    def read_egrid(cls, filename: str | os.PathLike) -> Self:
        return cls()
