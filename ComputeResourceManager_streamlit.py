"""
ComputeResourceManager.py

A module for managing compute resources for Ollama LLM operations.
Provides optimal settings based on available system resources.
"""

from typing import Dict
import psutil
import logging
from pathlib import Path

class ComputeResourceManager:
    """Manages compute resources for LLM operations."""
    
    def __init__(self):
        """Initialize compute resource manager with basic system detection."""
        self._setup_logging()
        # Get system resources
        self.cpu_count = psutil.cpu_count(logical=False)
        self.total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB

    def _setup_logging(self) -> None:
        """Configure basic logging for resource management."""
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_path / "compute_resources.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def test_system_details(self) -> None:
        """Display system configuration and resource availability."""
        print("\n=== System Configuration ===")
        print(f"CPU Cores: {self.cpu_count}")
        print(f"Total Memory: {self.total_memory:.1f} GB")
        print(f"Memory Available: {psutil.virtual_memory().available / (1024 * 1024 * 1024):.1f} GB")

    def get_compute_settings(self) -> Dict:
        """
        Determine optimal settings based on available system resources.
        
        Returns:
            Dict: Configuration settings for LLM operations
        """
        try:
            # Base settings for Ollama
            settings = {
                'temperature': 0.7,    # Slightly higher for more creative responses
                'top_k': 40,          # Balance between diversity and focus
                'top_p': 0.9,         # Higher value for more natural responses
                'num_thread': max(4, min(self.cpu_count * 2, 16)),  # Scale with CPU cores
                'context_window': 8192 # Default context window for most models
            }

            # Adjust based on available memory
            if self.total_memory >= 16:  # 16GB or more
                settings.update({
                    'max_tokens': 4096,
                    'context_window': 16384
                })
            elif self.total_memory >= 8:  # 8-16GB
                settings.update({
                    'max_tokens': 2048,
                    'context_window': 8192
                })
            else:  # Less than 8GB
                settings.update({
                    'max_tokens': 1024,
                    'context_window': 4096
                })

            self.logger.info(f"Computed settings: {settings}")
            return settings

        except Exception as e:
            self.logger.error(f"Error computing settings: {e}")
            # Fallback to conservative settings
            return {
                'temperature': 0.7,
                'top_k': 40,
                'top_p': 0.9,
                'num_thread': 4,
                'max_tokens': 1024,
                'context_window': 4096
            }

    def get_system_status(self) -> Dict:
        """
        Get current system resource usage.
        
        Returns:
            Dict: Current system resource metrics
        """
        try:
            memory = psutil.virtual_memory()
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_used_gb': (memory.total - memory.available) / (1024 * 1024 * 1024),
                'memory_percent': memory.percent
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {}