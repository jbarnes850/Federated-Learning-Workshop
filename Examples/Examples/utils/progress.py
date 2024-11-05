"""
Progress Tracking Utility
========================

This module provides consistent progress tracking and validation across all workshop components.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ProgressStep:
    name: str
    status: bool
    timestamp: datetime
    details: Optional[str] = None

class ProgressTracker:
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.steps: Dict[str, ProgressStep] = {}
        
    def mark_complete(self, step: str, details: Optional[str] = None) -> None:
        """Mark a step as complete with optional details."""
        self.steps[step] = ProgressStep(
            name=step,
            status=True,
            timestamp=datetime.now(),
            details=details
        )
        logger.info(f"✓ {step} completed")
        
    def mark_failed(self, step: str, error: str) -> None:
        """Mark a step as failed with error details."""
        self.steps[step] = ProgressStep(
            name=step,
            status=False,
            timestamp=datetime.now(),
            details=error
        )
        logger.error(f"❌ {step} failed: {error}")
        
    def get_progress(self) -> Dict[str, str]:
        """Get current progress status."""
        return {
            step.name: "✓" if step.status else "❌"
            for step in self.steps.values()
        } 