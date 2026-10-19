"""Kaggle account environment selection shared by the active pipeline."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def env_for_account(account: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("KAGGLE_CONFIG_DIR", None)
    env.pop("KAGGLE_API_TOKEN", None)
    if account == "vedanggggg":
        pass
    elif account == "vedangpandeyyy":
        fake_home = Path(tempfile.gettempdir()) / "kaggle_home_vedangpandeyyy"
        kdir = fake_home / ".kaggle"
        kdir.mkdir(parents=True, exist_ok=True)
        src = Path.home() / ".kaggle/kaggle.json"
        if src.exists():
            shutil.copy(str(src), str(kdir / "kaggle.json"))
            os.chmod(str(kdir / "kaggle.json"), 0o600)
        (kdir / "access_token").unlink(missing_ok=True)
        env["HOME"] = str(fake_home)
        env["KAGGLE_CONFIG_DIR"] = str(kdir)
    elif account in ("softmaxsimp", "samaltmannnn", "shoumikmitra"):
        token = (Path.home() / f".kaggle/profiles/{account}/access_token").read_text().strip()
        env["KAGGLE_API_TOKEN"] = token
    else:
        raise ValueError(account)
    return env
