"""Backward-compatible imports for behavior metrics.

New code should import running metrics from ``behavior.running`` and lick
metrics from ``behavior.licks``.
"""

from lamf_analysis.behavior.licks import add_lick_bouts
from lamf_analysis.behavior.running import (
    get_running_epochs,
    plot_running_speed_with_epochs,
    process_running_speed,
    total_run_distance,
)


__all__ = [
    "add_lick_bouts",
    "get_running_epochs",
    "plot_running_speed_with_epochs",
    "process_running_speed",
    "total_run_distance",
]
