"""
Runtime environment checks for solver scripts.

Call require_submodule() before any macro_place import.
It raises SubmoduleMissingError if the TILOS submodule is absent,
rather than letting a cryptic ModuleNotFoundError propagate.
Scripts should catch the exception, print it, and exit with code 1.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLC_MODULE = (
    _REPO_ROOT
    / "external"
    / "MacroPlacement"
    / "CodeElements"
    / "Plc_client"
    / "plc_client_os.py"
)


class SubmoduleMissingError(RuntimeError):
    """Raised when the TILOS MacroPlacement submodule is not initialized."""

    _MESSAGE = (
        "Error: TILOS MacroPlacement submodule not initialized.\n"
        "\n"
        "Initialize it with:\n"
        "    git submodule update --init external/MacroPlacement\n"
        "\n"
        "macro_place requires this submodule for benchmark loading,\n"
        "validation, and scoring."
    )

    def __init__(self) -> None:
        super().__init__(self._MESSAGE)


def require_submodule() -> None:
    """Raise SubmoduleMissingError if the TILOS MacroPlacement submodule is missing."""
    if not _PLC_MODULE.exists():
        raise SubmoduleMissingError()
