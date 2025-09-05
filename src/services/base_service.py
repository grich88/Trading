"""
Base Service

This module provides a base class for all services in the Trading Algorithm System.
"""

import gc
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import psutil

from src.config import Config
from src.utils.error_handling import AppError
from src.utils.logging_service import setup_logger
from src.utils.performance import performance_timer


class ServiceError(AppError):
    """Exception raised for service errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.

        Args:
            message: The error message.
            details: Additional error details.
        """
        super().__init__(message, 500, details)


class BaseService(ABC):
    """
    Base class for all services.

    This class provides common functionality for all services, including:
    - Lifecycle management (init, start, stop)
    - Error handling
    - Resource management
    - Memory monitoring
    - Health checks
    """

    def __init__(self, name: str):
        """
        Initialize the service.

        Args:
            name: The name of the service.
        """
        self.name = name
        self.logger = setup_logger(f"service.{name}")
        self.running = False
        self.memory_monitor_thread: Optional[threading.Thread] = None
        self.stop_memory_monitor = threading.Event()
        
        # Memory management settings
        self.memory_threshold = Config.get("MEMORY_THRESHOLD_PERCENT", 80)
        self.initial_batch_size = Config.get("INITIAL_BATCH_SIZE", 100)
        self.min_batch_size = Config.get("MIN_BATCH_SIZE", 10)
        self.max_batch_size = Config.get("MAX_BATCH_SIZE", 1000)
        self.gc_frequency = Config.get("GC_FREQUENCY", 5)
        self.memory_check_interval = Config.get("MEMORY_CHECK_INTERVAL_SECONDS", 60)
        self.enable_memory_monitoring = Config.get("ENABLE_MEMORY_MONITORING", True)
        
        # Current batch size for adaptive processing
        self.current_batch_size = self.initial_batch_size
        
        # Operation counter for garbage collection
        self.operation_count = 0

    def start(self) -> None:
        """
        Start the service.

        This method initializes resources and starts any background threads.

        Raises:
            ServiceError: If the service fails to start.
        """
        if self.running:
            self.logger.warning(f"Service {self.name} is already running")
            return

        try:
            self.logger.info(f"Starting service {self.name}")
            self._initialize_resources()
            self.running = True
            
            # Start memory monitoring if enabled
            if self.enable_memory_monitoring:
                self._start_memory_monitoring()
            
            self.logger.info(f"Service {self.name} started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start service {self.name}: {str(e)}")
            self.running = False
            raise ServiceError(f"Failed to start service {self.name}", {"error": str(e)})

    def stop(self) -> None:
        """
        Stop the service.

        This method cleans up resources and stops any background threads.

        Raises:
            ServiceError: If the service fails to stop.
        """
        if not self.running:
            self.logger.warning(f"Service {self.name} is not running")
            return

        try:
            self.logger.info(f"Stopping service {self.name}")
            
            # Stop memory monitoring
            if self.memory_monitor_thread and self.memory_monitor_thread.is_alive():
                self.stop_memory_monitor.set()
                self.memory_monitor_thread.join(timeout=5.0)
            
            self._cleanup_resources()
            self.running = False
            self.logger.info(f"Service {self.name} stopped successfully")
        except Exception as e:
            self.logger.error(f"Failed to stop service {self.name}: {str(e)}")
            raise ServiceError(f"Failed to stop service {self.name}", {"error": str(e)})

    def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the service.

        Returns:
            A dictionary containing health information.
        """
        return {
            "name": self.name,
            "status": "running" if self.running else "stopped",
            "memory_usage": self._get_memory_usage(),
        }

    def _initialize_resources(self) -> None:
        """
        Initialize resources for the service.

        This method should be overridden by subclasses to initialize
        any resources needed by the service.

        Raises:
            ServiceError: If resource initialization fails.
        """
        pass

    def _cleanup_resources(self) -> None:
        """
        Clean up resources used by the service.

        This method should be overridden by subclasses to clean up
        any resources used by the service.

        Raises:
            ServiceError: If resource cleanup fails.
        """
        pass

    def _get_memory_usage(self) -> Dict[str, float]:
        """
        Get the current memory usage.

        Returns:
            A dictionary containing memory usage information.
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
        }

    def _start_memory_monitoring(self) -> None:
        """
        Start memory monitoring in a background thread.
        """
        self.stop_memory_monitor.clear()
        self.memory_monitor_thread = threading.Thread(
            target=self._memory_monitor_task,
            daemon=True,
            name=f"{self.name}-memory-monitor",
        )
        self.memory_monitor_thread.start()

    def _memory_monitor_task(self) -> None:
        """
        Memory monitoring task that runs in a background thread.
        """
        self.logger.info(f"Memory monitoring started for service {self.name}")
        
        while not self.stop_memory_monitor.is_set():
            try:
                memory_usage = self._get_memory_usage()
                memory_percent = memory_usage["percent"]
                
                if memory_percent > self.memory_threshold:
                    self.logger.warning(
                        f"Memory usage above threshold: {memory_percent:.1f}% > {self.memory_threshold}%"
                    )
                    self._reduce_batch_size()
                    self._force_garbage_collection()
                elif memory_percent < self.memory_threshold * 0.7:
                    self._increase_batch_size()
                
                self.logger.debug(
                    f"Memory usage: {memory_percent:.1f}%, "
                    f"Batch size: {self.current_batch_size}"
                )
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {str(e)}")
            
            # Wait for the next check interval or until the stop event is set
            self.stop_memory_monitor.wait(self.memory_check_interval)
        
        self.logger.info(f"Memory monitoring stopped for service {self.name}")

    def _reduce_batch_size(self) -> None:
        """
        Reduce the batch size for adaptive processing.
        """
        new_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.7))
        
        if new_batch_size != self.current_batch_size:
            self.logger.info(
                f"Reducing batch size: {self.current_batch_size} -> {new_batch_size}"
            )
            self.current_batch_size = new_batch_size

    def _increase_batch_size(self) -> None:
        """
        Increase the batch size for adaptive processing.
        """
        new_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
        
        if new_batch_size != self.current_batch_size:
            self.logger.info(
                f"Increasing batch size: {self.current_batch_size} -> {new_batch_size}"
            )
            self.current_batch_size = new_batch_size

    def _force_garbage_collection(self) -> None:
        """
        Force garbage collection to free memory.
        """
        self.logger.info("Forcing garbage collection")
        gc.collect()

    @performance_timer
    def _increment_operation_count(self) -> None:
        """
        Increment the operation count and run garbage collection if needed.
        """
        self.operation_count += 1
        
        # Run garbage collection periodically
        if self.operation_count % self.gc_frequency == 0:
            self._force_garbage_collection()

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the service.

        Returns:
            A dictionary containing status information.
        """
        pass