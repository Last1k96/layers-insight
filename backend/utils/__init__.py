import re


def sanitize_filename(name: str) -> str:
    """Sanitize a string for safe use as a file/folder name on all platforms."""
    safe = re.sub(r'[^\w\-.]', '_', name)
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe or "unnamed"
