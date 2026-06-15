import json
from pathlib import Path
from typing import List

import pandas as pd

MAX_DISPLAY_POINTS = 1000

# how far from the end of the file to search for the last valid JSON line
_TAIL_CHUNK_BYTES = 4096


def parse_metrics_jsonl(log_path: Path, max_points: int = MAX_DISPLAY_POINTS) -> pd.DataFrame:
    """Read a JSONL metrics file and return a DataFrame.

    Each line is expected to be a JSON object with at least 'epoch' and 'loss'.
    Returns an empty DataFrame if the file does not exist or has no valid lines.
    """
    if not log_path.exists():
        return pd.DataFrame()

    records: List[dict] = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df.tail(max_points).reset_index(drop=True)


def latest_metrics(log_path: Path) -> dict:
    """Return the last valid JSON record from the JSONL file, or an empty dict.

    Reads only the last _TAIL_CHUNK_BYTES of the file instead of parsing the
    whole thing -- important when called every second during a long training run.
    """
    if not log_path.exists():
        return {}
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            if file_size == 0:
                return {}
            chunk = min(_TAIL_CHUNK_BYTES, file_size)
            f.seek(-chunk, 2)
            raw = f.read(chunk)
        lines = raw.decode("utf-8", errors="replace").splitlines()
        # scan from the end to find the last non-empty parseable line
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    except OSError:
        return {}
    return {}
