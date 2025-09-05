"""
Base service module.

This module provides a base class for all services with common functionality
such as logging, error handling, and memory management.
"""

import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime

# Import utilities
from src.utils import get_logger, MemoryMonitor, performance_monitor, exception_handler

# Import configuration
from src.config import (
    MEMORY_THRESHOLD_PERCENT,
    INITIAL_BATCH_SIZE,
    MIN_BATCH_SIZE,
    MAX_BATCH_SIZE,
    GC_FREQUENCY,
    MEMORY_CHECK_INTERVAL_SECONDS,
    ENABLE_MEMORY_MONITORING
)


class BaseService:
    """
    Base class for all services.
    
    This class provides common functionality for all services, including:
    - Logging
    - Error handling
    - Memory management
    - Lifecycle management (start, stop, health check)
    """
    
    def __init__(self, 
                 name: str,
                 enable_memory_monitoring: bool = ENABLE_MEMORY_MONITORING,
                 memory_threshold_percent: int = MEMORY_THRESHOLD_PERCENT,
                 memory_check_interval_seconds: int = MEMORY_CHECK_INTERVAL_SECONDS,
                 initial_batch_size: int = INITIAL_BATCH_SIZE,
                 min_batch_size: int = MIN_BATCH_SIZE,
                 max_batch_size: int = MAX_BATCH_SIZE,
                 gc_frequency: int = GC_FREQUENCY):
        """
        Initialize the base service.
        
        Args:
            name: Service name
            enable_memory_monitoring: Whether to enable memory monitoring
            memory_threshold_percent: Memory usage threshold as percentage
            memory_check_interval_seconds: Interval between memory checks in seconds
            initial_batch_size: Initial batch size for processing
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            gc_frequency: Garbage collection frequency (operations between GC)
        """
        self.name = name
        self.logger = get_logger(name)
        self.running = False
        self.start_time = None
        self.health_check_interval = 60  # seconds
        self.health_check_thread = None
        
        # Memory management
        self.enable_memory_monitoring = enable_memory_monitoring
        if enable_memory_monitoring:
            self.memory_monitor = MemoryMonitor(
                threshold_percent=memory_threshold_percent,
                check_interval_seconds=memory_check_interval_seconds,
                initial_batch_size=initial_batch_size,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                gc_frequency=gc_frequency
            )
        else:
            self.memory_monitor = None
        
        # Health metrics
        self.health_metrics = {
            "status": "initialized",
            "uptime_seconds": 0,
            "last_health_check": None,
            "error_count": 0,
            "warning_count": 0,
            "memory_metrics": {}
        }
        
        self.logger.info(f"{self.name} service initialized")
    
    def start(self) -> bool:
        """
        Start the service.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            self.logger.warning(f"{self.name} service is already running")
            return True
        
        try:
            self.logger.info(f"Starting {self.name} service")
            
            # Start memory monitoring if enabled
            if self.enable_memory_monitoring and self.memory_monitor:
                self.memory_monitor.start_monitoring()
            
            # Set service state
            self.running = True
            self.start_time = datetime.now()
            self.health_metrics["status"] = "running"
            
            # Start health check thread
            self._start_health_check_thread()
            
            # Call service-specific start method
            self._start_service()
            
            self.logger.info(f"{self.name} service started successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error starting {self.name} service: {str(e)}")
            self.health_metrics["status"] = "error"
            self.health_metrics["error_count"] += 1
            return False
    
    def stop(self) -> bool:
        """
        Stop the service.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.running:
            self.logger.warning(f"{self.name} service is not running")
            return True
        
        try:
            self.logger.info(f"Stopping {self.name} service")
            
            # Set service state
            self.running = False
            self.health_metrics["status"] = "stopped"
            
            # Stop health check thread
            if self.health_check_thread and self.health_check_thread.is_alive():
                self.health_check_thread.join(timeout=1.0)
            
            # Stop memory monitoring if enabled
            if self.enable_memory_monitoring and self.memory_monitor:
                self.memory_monitor.stop_monitoring()
            
            # Call service-specific stop method
            self._stop_service()
            
            self.logger.info(f"{self.name} service stopped successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error stopping {self.name} service: {str(e)}")
            self.health_metrics["status"] = "error"
            self.health_metrics["error_count"] += 1
            return False
    
    def _start_service(self) -> None:
        """
        Service-specific start logic.
        
        This method should be overridden by subclasses to implement
        service-specific start logic.
        """
        pass
    
    def _stop_service(self) -> None:
        """
        Service-specific stop logic.
        
        This method should be overridden by subclasses to implement
        service-specific stop logic.
        """
        pass
    
    def _start_health_check_thread(self) -> None:
        """Start the health check thread."""
        if not self.running:
            return
        
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
    
    def _health_check_loop(self) -> None:
        """Health check thread function."""
        while self.running:
            try:
                self.check_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in health check: {str(e)}")
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check service health.
        
        Returns:
            Dictionary with health metrics
        """
        try:
            # Update uptime
            if self.start_time:
                uptime = (datetime.now() - self.start_time).total_seconds()
                self.health_metrics["uptime_seconds"] = uptime
            
            # Update memory metrics if monitoring is enabled
            if self.enable_memory_monitoring and self.memory_monitor:
                self.health_metrics["memory_metrics"] = self.memory_monitor.get_metrics()
            
            # Update last health check time
            self.health_metrics["last_health_check"] = datetime.now()
            
            # Call service-specific health check
            service_health = self._check_service_health()
            if service_health:
                self.health_metrics.update(service_health)
            
            # Log health check
            self.logger.debug(f"Health check: {self.health_metrics['status']}")
            
            return self.health_metrics
        
        except Exception as e:
            self.logger.error(f"Error in health check: {str(e)}")
            self.health_metrics["status"] = "error"
            self.health_metrics["error_count"] += 1
            return self.health_metrics
    
    def _check_service_health(self) -> Optional[Dict[str, Any]]:
        """
        Service-specific health check logic.
        
        This method should be overridden by subclasses to implement
        service-specific health check logic.
        
        Returns:
            Dictionary with service-specific health metrics, or None
        """
        return None
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get current health metrics.
        
        Returns:
            Dictionary with health metrics
        """
        return self.health_metrics
    
    def process_batch(self, 
                     items: List[Any], 
                     process_func: Callable[[List[Any]], Any],
                     batch_size: Optional[int] = None) -> List[Any]:
        """
        Process items in adaptive batches to manage memory usage.
        
        Args:
            items: List of items to process
            process_func: Function to process a batch of items
            batch_size: Initial batch size (uses monitor's batch size if None)
            
        Returns:
            List of results from processing
        """
        if not self.enable_memory_monitoring or not self.memory_monitor:
            # Process all items at once if memory monitoring is disabled
            return process_func(items)
        
        results = []
        total_items = len(items)
        processed = 0
        
        if batch_size is None:
            batch_size = self.memory_monitor.get_batch_size()
        
        while processed < total_items:
            # Get current batch size from monitor
            current_batch_size = self.memory_monitor.get_batch_size()
            
            # Process batch
            end_idx = min(processed + current_batch_size, total_items)
            batch = items[processed:end_idx]
            batch_results = process_func(batch)
            
            # Add results
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
            
            # Update processed count
            batch_processed = end_idx - processed
            processed += batch_processed
            
            # Log progress
            self.logger.debug(
                f"Processed {processed}/{total_items} items "
                f"({processed/total_items*100:.1f}%) with batch size {current_batch_size}"
            )
            
            # Maybe collect garbage
            self.memory_monitor.maybe_collect_garbage()
        
        return results
    
    @performance_monitor()
    def execute_with_monitoring(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function with performance monitoring.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
        """
        return func(*args, **kwargs)
    
    @exception_handler(log_exception=True)
    def execute_with_error_handling(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function with error handling.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function, or None if an error occurs
        """
        return func(*args, **kwargs)


class LongRunningService(BaseService):
    """
    Base class for long-running services.
    
    This class extends BaseService with additional functionality for
    long-running services, such as worker threads and task queues.
    """
    
    def __init__(self, 
                 name: str,
                 worker_count: int = 1,
                 **kwargs: Any):
        """
        Initialize the long-running service.
        
        Args:
            name: Service name
            worker_count: Number of worker threads
            **kwargs: Additional arguments for BaseService
        """
        super().__init__(name, **kwargs)
        
        self.worker_count = worker_count
        self.worker_threads = []
        self.task_queue = []
        self.queue_lock = threading.Lock()
    
    def _start_service(self) -> None:
        """Start worker threads."""
        self.worker_threads = []
        for i in range(self.worker_count):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"{self.name}-worker-{i}",
                daemon=True
            )
            self.worker_threads.append(thread)
            thread.start()
        
        self.logger.info(f"Started {self.worker_count} worker threads")
    
    def _stop_service(self) -> None:
        """Stop worker threads."""
        # Wait for worker threads to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        self.worker_threads = []
        self.logger.info("Worker threads stopped")
    
    def _worker_loop(self) -> None:
        """Worker thread function."""
        thread_name = threading.current_thread().name
        self.logger.debug(f"Worker thread {thread_name} started")
        
        while self.running:
            try:
                # Get next task from queue
                task = self._get_next_task()
                
                if task:
                    # Process task
                    self._process_task(task)
                else:
                    # No tasks available, sleep
                    time.sleep(0.1)
            
            except Exception as e:
                self.logger.error(f"Error in worker thread {thread_name}: {str(e)}")
                self.health_metrics["error_count"] += 1
        
        self.logger.debug(f"Worker thread {thread_name} stopped")
    
    def _get_next_task(self) -> Optional[Any]:
        """
        Get the next task from the queue.
        
        Returns:
            Next task, or None if queue is empty
        """
        with self.queue_lock:
            if self.task_queue:
                return self.task_queue.pop(0)
        return None
    
    def _process_task(self, task: Any) -> None:
        """
        Process a task.
        
        This method should be overridden by subclasses to implement
        task processing logic.
        
        Args:
            task: Task to process
        """
        pass
    
    def add_task(self, task: Any) -> None:
        """
        Add a task to the queue.
        
        Args:
            task: Task to add
        """
        with self.queue_lock:
            self.task_queue.append(task)
    
    def get_queue_size(self) -> int:
        """
        Get the current queue size.
        
        Returns:
            Number of tasks in the queue
        """
        with self.queue_lock:
            return len(self.task_queue)
    
    def _check_service_health(self) -> Dict[str, Any]:
        """
        Check service-specific health.
        
        Returns:
            Dictionary with service-specific health metrics
        """
        return {
            "worker_count": self.worker_count,
            "active_workers": sum(1 for t in self.worker_threads if t.is_alive()),
            "queue_size": self.get_queue_size()
        }


if __name__ == "__main__":
    # Example usage
    
    # Create a simple service
    service = BaseService("ExampleService")
    service.start()
    
    # Check health
    health = service.check_health()
    print(f"Service health: {health}")
    
    # Stop service
    service.stop()
    
    # Create a long-running service
    class ExampleLongRunningService(LongRunningService):
        def _process_task(self, task):
            print(f"Processing task: {task}")
            time.sleep(0.5)  # Simulate processing
    
    long_service = ExampleLongRunningService("ExampleLongRunningService", worker_count=2)
    long_service.start()
    
    # Add some tasks
    for i in range(5):
        long_service.add_task(f"Task {i}")
    
    # Wait for tasks to be processed
    time.sleep(3)
    
    # Check health
    health = long_service.check_health()
    print(f"Long-running service health: {health}")
    
    # Stop service
    long_service.stop()
