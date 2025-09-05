"""
Performance monitoring module.

This module provides utilities for monitoring and optimizing performance.
"""

import time
import gc
import psutil
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast, Union
from datetime import datetime

# Import logging service
from src.utils.logging_service import get_logger

# Create logger
logger = get_logger("Performance")

# Type variables for function annotations
F = TypeVar('F', bound=Callable[..., Any])


def performance_monitor(operation_name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator for monitoring function performance.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get operation name
            op_name = operation_name or func.__name__
            
            # Record start time and memory
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Calculate elapsed time and memory usage
            elapsed_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            memory_diff = end_memory - start_memory
            
            # Log performance metrics
            logger.info(
                f"Performance: {op_name} took {elapsed_time:.4f} seconds, "
                f"memory change: {memory_diff:.2f} MB"
            )
            
            return result
        
        return cast(F, wrapper)
    
    return decorator


class MemoryMonitor:
    """
    Memory monitoring utility for long-running processes.
    
    This class provides memory monitoring capabilities for long-running processes,
    with automatic garbage collection and adaptive batch processing.
    """
    
    def __init__(self, 
                 threshold_percent: int = 80,
                 check_interval_seconds: int = 60,
                 initial_batch_size: int = 100,
                 min_batch_size: int = 10,
                 max_batch_size: int = 1000,
                 gc_frequency: int = 5):
        """
        Initialize the memory monitor.
        
        Args:
            threshold_percent: Memory usage threshold as percentage
            check_interval_seconds: Interval between memory checks in seconds
            initial_batch_size: Initial batch size for processing
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            gc_frequency: Garbage collection frequency (operations between GC)
        """
        self.threshold_percent = threshold_percent
        self.check_interval_seconds = check_interval_seconds
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.gc_frequency = gc_frequency
        
        self.current_batch_size = initial_batch_size
        self.operation_count = 0
        self.monitoring = False
        self.monitor_thread = None
        
        # Performance metrics
        self.metrics = {
            "peak_memory_mb": 0.0,
            "current_memory_mb": 0.0,
            "memory_percent": 0.0,
            "gc_collections": 0,
            "batch_size_adjustments": 0,
            "last_check_time": None
        }
    
    def start_monitoring(self) -> None:
        """Start the memory monitoring thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the memory monitoring thread."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_memory(self) -> None:
        """Memory monitoring thread function."""
        while self.monitoring:
            self._check_memory()
            time.sleep(self.check_interval_seconds)
    
    def _check_memory(self) -> Dict[str, Any]:
        """
        Check current memory usage and adjust batch size if needed.
        
        Returns:
            Dictionary with memory metrics
        """
        # Get memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # MB
        
        # Get system memory usage
        system_memory = psutil.virtual_memory()
        memory_percent = system_memory.percent
        
        # Update metrics
        self.metrics["current_memory_mb"] = memory_mb
        self.metrics["memory_percent"] = memory_percent
        self.metrics["last_check_time"] = datetime.now()
        
        if memory_mb > self.metrics["peak_memory_mb"]:
            self.metrics["peak_memory_mb"] = memory_mb
        
        # Log memory usage
        logger.debug(
            f"Memory usage: {memory_mb:.2f} MB ({memory_percent:.1f}%), "
            f"batch size: {self.current_batch_size}"
        )
        
        # Adjust batch size if memory usage is above threshold
        if memory_percent > self.threshold_percent:
            old_batch_size = self.current_batch_size
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.75)
            )
            
            if old_batch_size != self.current_batch_size:
                self.metrics["batch_size_adjustments"] += 1
                logger.warning(
                    f"Memory usage above threshold ({memory_percent:.1f}% > {self.threshold_percent}%), "
                    f"reducing batch size from {old_batch_size} to {self.current_batch_size}"
                )
                
                # Force garbage collection
                gc.collect()
                self.metrics["gc_collections"] += 1
        
        # Increase batch size if memory usage is well below threshold
        elif memory_percent < (self.threshold_percent * 0.7):
            old_batch_size = self.current_batch_size
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.25)
            )
            
            if old_batch_size != self.current_batch_size:
                self.metrics["batch_size_adjustments"] += 1
                logger.debug(
                    f"Memory usage well below threshold ({memory_percent:.1f}% < {self.threshold_percent * 0.7:.1f}%), "
                    f"increasing batch size from {old_batch_size} to {self.current_batch_size}"
                )
        
        return self.metrics
    
    def get_batch_size(self) -> int:
        """
        Get the current recommended batch size.
        
        Returns:
            Current batch size
        """
        return self.current_batch_size
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current memory metrics.
        
        Returns:
            Dictionary with memory metrics
        """
        # Update metrics before returning
        self._check_memory()
        return self.metrics
    
    def maybe_collect_garbage(self) -> None:
        """Run garbage collection periodically based on operation count."""
        self.operation_count += 1
        
        if self.operation_count % self.gc_frequency == 0:
            gc.collect()
            self.metrics["gc_collections"] += 1
            logger.debug(f"Garbage collection performed (operation {self.operation_count})")


# Create global memory monitor instance
memory_monitor = MemoryMonitor()


def get_memory_monitor() -> MemoryMonitor:
    """
    Get the global memory monitor instance.
    
    Returns:
        The global memory monitor instance
    """
    return memory_monitor


def adaptive_batch_processing(
    items: List[Any],
    process_func: Callable[[List[Any]], Any],
    monitor: Optional[MemoryMonitor] = None,
    batch_size: Optional[int] = None
) -> List[Any]:
    """
    Process items in adaptive batches to manage memory usage.
    
    Args:
        items: List of items to process
        process_func: Function to process a batch of items
        monitor: Memory monitor instance (uses global instance if None)
        batch_size: Initial batch size (uses monitor's batch size if None)
        
    Returns:
        List of results from processing
    """
    if monitor is None:
        monitor = memory_monitor
    
    if batch_size is None:
        batch_size = monitor.get_batch_size()
    
    results = []
    total_items = len(items)
    processed = 0
    
    while processed < total_items:
        # Get current batch size from monitor
        current_batch_size = monitor.get_batch_size()
        
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
        logger.debug(
            f"Processed {processed}/{total_items} items "
            f"({processed/total_items*100:.1f}%) with batch size {current_batch_size}"
        )
        
        # Maybe collect garbage
        monitor.maybe_collect_garbage()
    
    return results


if __name__ == "__main__":
    # Example usage
    
    # Using the performance monitor decorator
    @performance_monitor("example_operation")
    def process_data(data_size: int) -> List[int]:
        # Simulate data processing
        result = []
        for i in range(data_size):
            result.append(i * i)
        return result
    
    # Test performance monitor
    print("Testing performance monitor:")
    result = process_data(1000000)
    print(f"Processed {len(result)} items")
    
    # Test memory monitor
    print("\nTesting memory monitor:")
    monitor = MemoryMonitor(threshold_percent=70)
    monitor.start_monitoring()
    
    # Simulate batch processing
    def process_batch(batch: List[int]) -> List[int]:
        return [x * 2 for x in batch]
    
    items = list(range(10000))
    results = adaptive_batch_processing(items, process_batch, monitor)
    print(f"Processed {len(results)} items with adaptive batching")
    
    # Print memory metrics
    metrics = monitor.get_metrics()
    print(f"Memory usage: {metrics['current_memory_mb']:.2f} MB")
    print(f"Peak memory: {metrics['peak_memory_mb']:.2f} MB")
    print(f"Memory percent: {metrics['memory_percent']:.1f}%")
    print(f"Garbage collections: {metrics['gc_collections']}")
    print(f"Batch size adjustments: {metrics['batch_size_adjustments']}")
    
    monitor.stop_monitoring()
