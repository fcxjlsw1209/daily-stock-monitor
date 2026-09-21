from pathlib import Path
import os


def runtime_dir():
    return Path(os.environ.get("STOCK_MONITOR_HOME", Path.cwd() / ".stock-monitor")).expanduser().resolve()
