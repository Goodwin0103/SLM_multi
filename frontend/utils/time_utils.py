"""Utility helpers for time/duration formatting."""


def fmt_seconds(sec: float) -> str:
    """Convert a duration in seconds to a compact human-readable string.

    Examples:
        3661 -> "1h 01m 01s"
        125  -> "2m 05s"
        45   -> "45s"
    """
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"
