"""
Progress Tracking Utilities
=========================

This module provides utilities for tracking and logging progress during model training
and deployment.
"""

import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressStatus(Enum):
    """Status indicators for progress tracking."""
    NOT_STARTED = "⏳"
    IN_PROGRESS = "🔄"
    COMPLETE = "✓"
    FAILED = "❌"

def log_progress(step: str, status: ProgressStatus) -> None:
    """
    Log progress status with consistent formatting.
    
    Args:
        step: Name of the step being tracked
        status: Current status from ProgressStatus enum
    """
    status_icon = status.value
    logger.info(f"{status_icon} {step}")