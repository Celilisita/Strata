"""Validate the explicitly selected task temp root on Windows hosts."""
import os
from pathlib import Path


def require_task_temp_dir(directory, family):
    if os.name != "nt":
        return
    root = Path(os.environ.get("STRATA_TEMP_ROOT", "Y:/temp")).resolve()
    if (root.drive.upper() not in {"D:", "G:", "Y:"} or len(root.parts) < 2 or
            root.parts[1].casefold() != "temp"):
        raise ValueError("STRATA_TEMP_ROOT must select an authorized drive temp task directory")
    if not Path(directory).resolve().is_relative_to(root):
        raise ValueError(f"{family} temporary artifacts must stay below {root}")
