"""
Performance monitoring module.

This module provides utilities for monitoring and optimizing performance,
including function timing, memory usage tracking, adaptive batch processing,
and profiling tools.
"""

import time
import gc
import psutil
import functools
import threading
import asyncio
import cProfile
import pstats
import io
import tempfile
import os
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast, Union, Awaitable, Tuple
from datetime import datetime
from contextlib import contextmanager

# Import logging service
from src.utils.logging_service import get_logger
from src.utils.error_handling import error_context

# Create logger
logger = get_logger("Performance")

# Type variables for function annotations
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Awaitable[Any]])
T = TypeVar('T')


def performance_monitor(operation_name: Optional[str] = None, log_level: str = "INFO") -> Callable[[F], F]:
    """
    Decorator for monitoring function performance.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        log_level: Log level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
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
            log_method = getattr(logger, log_level.lower(), logger.info)
            log_method(
                f"Performance: {op_name} took {elapsed_time:.4f} seconds, "
                f"memory change: {memory_diff:.2f} MB"
            )
            
            return result
        
        return cast(F, wrapper)
    
    return decorator


def async_performance_monitor(operation_name: Optional[str] = None, log_level: str = "INFO") -> Callable[[AsyncF], AsyncF]:
    """
    Decorator for monitoring async function performance.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        log_level: Log level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Decorated async function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get operation name
            op_name = operation_name or func.__name__
            
            # Record start time and memory
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            # Call the function
            result = await func(*args, **kwargs)
            
            # Calculate elapsed time and memory usage
            elapsed_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            memory_diff = end_memory - start_memory
            
            # Log performance metrics
            log_method = getattr(logger, log_level.lower(), logger.info)
            log_method(
                f"Performance: {op_name} took {elapsed_time:.4f} seconds, "
                f"memory change: {memory_diff:.2f} MB"
            )
            
            return result
        
        return cast(AsyncF, wrapper)
    
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


@contextmanager
def timer(operation_name: str, log_level: str = "INFO") -> None:
    """
    Context manager for timing operations.
    
    Args:
        operation_name: Name of the operation
        log_level: Log level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Yields:
        None
    """
    # Record start time
    start_time = time.time()
    
    try:
        yield
    finally:
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Log performance metrics
        log_method = getattr(logger, log_level.lower(), logger.info)
        log_method(f"Timer: {operation_name} took {elapsed_time:.4f} seconds")


@contextmanager
def memory_usage(operation_name: str, log_level: str = "INFO") -> None:
    """
    Context manager for tracking memory usage.
    
    Args:
        operation_name: Name of the operation
        log_level: Log level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Yields:
        None
    """
    # Record start memory
    process = psutil.Process()
    start_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    try:
        yield
    finally:
        # Calculate memory usage
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_memory - start_memory
        
        # Log performance metrics
        log_method = getattr(logger, log_level.lower(), logger.info)
        log_method(f"Memory: {operation_name} used {memory_diff:.2f} MB")


@contextmanager
def profiler(operation_name: str, save_path: Optional[str] = None, print_stats: bool = False) -> None:
    """
    Context manager for profiling operations.
    
    Args:
        operation_name: Name of the operation
        save_path: Path to save profiling results (optional)
        print_stats: Whether to print profiling stats to console
        
    Yields:
        None
    """
    # Create profiler
    pr = cProfile.Profile()
    
    # Start profiling
    pr.enable()
    
    try:
        yield
    finally:
        # Stop profiling
        pr.disable()
        
        # Create string buffer for stats
        s = io.StringIO()
        
        # Create stats object
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        
        # Print stats to buffer
        ps.print_stats(30)  # Top 30 functions
        
        # Get stats string
        stats_str = s.getvalue()
        
        # Log summary
        logger.info(f"Profiling results for {operation_name}:")
        
        # Print stats if requested
        if print_stats:
            print(f"\nProfiling results for {operation_name}:")
            print(stats_str)
        
        # Save stats if requested
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save stats to file
            with open(save_path, 'w') as f:
                f.write(stats_str)
            
            logger.info(f"Profiling results saved to {save_path}")


def adaptive_batch_processing(
    items: List[Any],
    process_func: Callable[[List[Any]], Any],
    monitor: Optional[MemoryMonitor] = None,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Any]:
    """
    Process items in adaptive batches to manage memory usage.
    
    Args:
        items: List of items to process
        process_func: Function to process a batch of items
        monitor: Memory monitor instance (uses global instance if None)
        batch_size: Initial batch size (uses monitor's batch size if None)
        progress_callback: Optional callback function for progress updates
        
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
        
        with error_context(f"Error processing batch {processed}-{end_idx}"):
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
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback(processed, total_items)
        
        # Maybe collect garbage
        monitor.maybe_collect_garbage()
    
    return results


async def async_adaptive_batch_processing(
    items: List[Any],
    process_func: Callable[[List[Any]], Awaitable[Any]],
    monitor: Optional[MemoryMonitor] = None,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Any]:
    """
    Process items in adaptive batches to manage memory usage (async version).
    
    Args:
        items: List of items to process
        process_func: Async function to process a batch of items
        monitor: Memory monitor instance (uses global instance if None)
        batch_size: Initial batch size (uses monitor's batch size if None)
        progress_callback: Optional callback function for progress updates
        
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
        
        try:
            batch_results = await process_func(batch)
        except Exception as e:
            logger.error(f"Error processing batch {processed}-{end_idx}: {str(e)}")
            raise
        
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
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback(processed, total_items)
        
        # Maybe collect garbage
        monitor.maybe_collect_garbage()
    
    return results


class SystemProfiler:
    """
    System resource profiler for monitoring CPU and memory usage.
    
    This class provides utilities for monitoring CPU and memory usage
    over time, with support for periodic sampling and reporting.
    """
    
    def __init__(self, 
                 sample_interval_seconds: float = 1.0,
                 log_level: str = "DEBUG",
                 max_samples: int = 3600):  # Default to 1 hour of samples at 1 second interval
        """
        Initialize the system profiler.
        
        Args:
            sample_interval_seconds: Interval between samples in seconds
            log_level: Log level to use for reporting
            max_samples: Maximum number of samples to store
        """
        self.sample_interval = sample_interval_seconds
        self.log_level = log_level
        self.max_samples = max_samples
        
        self.monitoring = False
        self.monitor_thread = None
        
        # Sample storage
        self.samples = {
            "timestamps": [],
            "cpu_percent": [],
            "memory_percent": [],
            "memory_mb": []
        }
        
        # Statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "avg_cpu_percent": 0,
            "max_cpu_percent": 0,
            "avg_memory_percent": 0,
            "max_memory_percent": 0,
            "avg_memory_mb": 0,
            "max_memory_mb": 0,
            "sample_count": 0
        }
    
    def start_monitoring(self) -> None:
        """Start monitoring system resources."""
        if self.monitoring:
            return
        
        # Reset samples and stats
        self._reset()
        
        # Set start time
        self.stats["start_time"] = datetime.now()
        
        # Start monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
        logger.debug("System resource monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop monitoring system resources.
        
        Returns:
            Dictionary with statistics
        """
        if not self.monitoring:
            return self.stats
        
        # Stop monitoring
        self.monitoring = False
        
        # Wait for thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        # Set end time
        self.stats["end_time"] = datetime.now()
        
        # Calculate duration
        if self.stats["start_time"]:
            self.stats["duration_seconds"] = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        # Calculate statistics
        self._calculate_stats()
        
        # Log summary
        log_method = getattr(logger, self.log_level.lower(), logger.debug)
        log_method(
            f"System monitoring summary: "
            f"CPU: {self.stats['avg_cpu_percent']:.1f}% avg, {self.stats['max_cpu_percent']:.1f}% max, "
            f"Memory: {self.stats['avg_memory_mb']:.1f} MB avg, {self.stats['max_memory_mb']:.1f} MB max"
        )
        
        return self.stats
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.
        
        Returns:
            Dictionary with statistics
        """
        # Calculate statistics
        self._calculate_stats()
        return self.stats
    
    def get_samples(self) -> Dict[str, List[Any]]:
        """
        Get all samples.
        
        Returns:
            Dictionary with sample lists
        """
        return self.samples
    
    def _reset(self) -> None:
        """Reset samples and statistics."""
        # Reset samples
        self.samples = {
            "timestamps": [],
            "cpu_percent": [],
            "memory_percent": [],
            "memory_mb": []
        }
        
        # Reset statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "avg_cpu_percent": 0,
            "max_cpu_percent": 0,
            "avg_memory_percent": 0,
            "max_memory_percent": 0,
            "avg_memory_mb": 0,
            "max_memory_mb": 0,
            "sample_count": 0
        }
    
    def _monitor_resources(self) -> None:
        """Monitor system resources in a separate thread."""
        while self.monitoring:
            # Take a sample
            self._take_sample()
            
            # Sleep for the sample interval
            time.sleep(self.sample_interval)
    
    def _take_sample(self) -> None:
        """Take a sample of system resources."""
        # Get current timestamp
        timestamp = datetime.now()
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent()
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / (1024 * 1024)  # MB
        
        # Add sample to storage
        self.samples["timestamps"].append(timestamp)
        self.samples["cpu_percent"].append(cpu_percent)
        self.samples["memory_percent"].append(memory_percent)
        self.samples["memory_mb"].append(memory_mb)
        
        # Limit sample storage
        if len(self.samples["timestamps"]) > self.max_samples:
            self.samples["timestamps"].pop(0)
            self.samples["cpu_percent"].pop(0)
            self.samples["memory_percent"].pop(0)
            self.samples["memory_mb"].pop(0)
    
    def _calculate_stats(self) -> None:
        """Calculate statistics from samples."""
        # Get number of samples
        sample_count = len(self.samples["timestamps"])
        self.stats["sample_count"] = sample_count
        
        if sample_count == 0:
            return
        
        # Calculate CPU statistics
        self.stats["avg_cpu_percent"] = sum(self.samples["cpu_percent"]) / sample_count
        self.stats["max_cpu_percent"] = max(self.samples["cpu_percent"])
        
        # Calculate memory statistics
        self.stats["avg_memory_percent"] = sum(self.samples["memory_percent"]) / sample_count
        self.stats["max_memory_percent"] = max(self.samples["memory_percent"])
        self.stats["avg_memory_mb"] = sum(self.samples["memory_mb"]) / sample_count
        self.stats["max_memory_mb"] = max(self.samples["memory_mb"])


def run_with_profiling(
    func: Callable[..., T],
    *args: Any,
    operation_name: Optional[str] = None,
    profile_cpu: bool = True,
    profile_memory: bool = True,
    save_path: Optional[str] = None,
    **kwargs: Any
) -> Tuple[T, Dict[str, Any]]:
    """
    Run a function with profiling.
    
    Args:
        func: Function to run
        *args: Arguments to pass to the function
        operation_name: Name of the operation (defaults to function name)
        profile_cpu: Whether to profile CPU usage
        profile_memory: Whether to profile memory usage
        save_path: Path to save profiling results (optional)
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Tuple of (function result, profiling stats)
    """
    # Get operation name
    op_name = operation_name or func.__name__
    
    # Create profilers
    system_profiler = SystemProfiler() if profile_cpu or profile_memory else None
    
    # Start profiling
    if system_profiler:
        system_profiler.start_monitoring()
    
    # Run the function with profiling
    with profiler(op_name, save_path=save_path):
        result = func(*args, **kwargs)
    
    # Stop profiling
    stats = {}
    if system_profiler:
        stats = system_profiler.stop_monitoring()
    
    return result, stats


async def run_with_async_profiling(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    operation_name: Optional[str] = None,
    profile_cpu: bool = True,
    profile_memory: bool = True,
    save_path: Optional[str] = None,
    **kwargs: Any
) -> Tuple[T, Dict[str, Any]]:
    """
    Run an async function with profiling.
    
    Args:
        func: Async function to run
        *args: Arguments to pass to the function
        operation_name: Name of the operation (defaults to function name)
        profile_cpu: Whether to profile CPU usage
        profile_memory: Whether to profile memory usage
        save_path: Path to save profiling results (optional)
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Tuple of (function result, profiling stats)
    """
    # Get operation name
    op_name = operation_name or func.__name__
    
    # Create profilers
    system_profiler = SystemProfiler() if profile_cpu or profile_memory else None
    
    # Start profiling
    if system_profiler:
        system_profiler.start_monitoring()
    
    # Run the function with profiling
    with profiler(op_name, save_path=save_path):
        result = await func(*args, **kwargs)
    
    # Stop profiling
    stats = {}
    if system_profiler:
        stats = system_profiler.stop_monitoring()
    
    return result, stats


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
    
    # Test context managers
    print("\nTesting context managers:")
    
    with timer("test_operation"):
        # Simulate work
        time.sleep(1)
    
    with memory_usage("memory_test"):
        # Simulate memory usage
        big_list = [0] * 1000000
    
    # Test profiler
    print("\nTesting profiler:")
    
    with profiler("profiler_test", print_stats=True):
        # Simulate work
        result = 0
        for i in range(1000000):
            result += i
    
    # Test run_with_profiling
    print("\nTesting run_with_profiling:")
    
    def cpu_intensive_task(n: int) -> int:
        result = 0
        for i in range(n):
            result += i
        return result
    
    result, stats = run_with_profiling(cpu_intensive_task, 1000000, operation_name="cpu_test")
    
    print(f"Result: {result}")
    print(f"CPU: {stats['avg_cpu_percent']:.1f}% avg, {stats['max_cpu_percent']:.1f}% max")
    print(f"Memory: {stats['avg_memory_mb']:.1f} MB avg, {stats['max_memory_mb']:.1f} MB max")
