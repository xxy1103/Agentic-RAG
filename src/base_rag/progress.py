from __future__ import annotations


def print_task_progress(stage: str, completed: int, total: int, detail: str = "") -> None:
    """Render one compact, terminal-friendly progress bar for a bounded task."""
    width = 24
    filled = int(width * completed / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" 当前:{detail}" if detail else ""
    print(f"\r[{stage:<14}] [{bar}] {completed:>4}/{total}{suffix}", end="", flush=True)
    if completed >= total:
        print(flush=True)
