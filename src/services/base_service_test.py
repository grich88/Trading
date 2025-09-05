"""
Tests for the base service module.
"""

import threading
import time
from unittest import mock

import pytest

from src.services.base_service import BaseService, ServiceError


class TestService(BaseService):
    """Test service implementation for testing."""
    
    def __init__(self, name="test"):
        super().__init__(name)
        self.initialized = False
        self.cleaned_up = False
        self.status = "idle"
    
    def _initialize_resources(self):
        self.initialized = True
        self.status = "initialized"
    
    def _cleanup_resources(self):
        self.cleaned_up = True
        self.status = "cleaned_up"
    
    def get_status(self):
        return {"status": self.status}


def test_service_lifecycle():
    """Test service lifecycle (init, start, stop)."""
    # Create service
    service = TestService()
    
    # Check initial state
    assert service.name == "test"
    assert service.running is False
    assert service.initialized is False
    assert service.cleaned_up is False
    
    # Start service
    service.start()
    
    # Check running state
    assert service.running is True
    assert service.initialized is True
    assert service.cleaned_up is False
    
    # Stop service
    service.stop()
    
    # Check stopped state
    assert service.running is False
    assert service.initialized is True
    assert service.cleaned_up is True


def test_service_start_error():
    """Test error handling during service start."""
    # Create service with error in initialization
    service = TestService()
    
    # Mock _initialize_resources to raise an exception
    with mock.patch.object(service, '_initialize_resources', side_effect=Exception("Test error")):
        # Start service should raise ServiceError
        with pytest.raises(ServiceError):
            service.start()
        
        # Service should not be running
        assert service.running is False


def test_service_stop_error():
    """Test error handling during service stop."""
    # Create and start service
    service = TestService()
    service.start()
    
    # Mock _cleanup_resources to raise an exception
    with mock.patch.object(service, '_cleanup_resources', side_effect=Exception("Test error")):
        # Stop service should raise ServiceError
        with pytest.raises(ServiceError):
            service.stop()


def test_service_health_check():
    """Test service health check."""
    # Create service
    service = TestService()
    
    # Check health before starting
    health = service.check_health()
    assert health["name"] == "test"
    assert health["status"] == "stopped"
    assert "memory_usage" in health
    
    # Start service
    service.start()
    
    # Check health after starting
    health = service.check_health()
    assert health["name"] == "test"
    assert health["status"] == "running"
    assert "memory_usage" in health
    
    # Stop service
    service.stop()


def test_memory_monitoring():
    """Test memory monitoring functionality."""
    # Create service with short memory check interval
    service = TestService()
    service.memory_check_interval = 0.1
    
    # Start service
    service.start()
    
    # Check that memory monitor thread is running
    assert service.memory_monitor_thread is not None
    assert service.memory_monitor_thread.is_alive()
    
    # Wait for a few memory check intervals
    time.sleep(0.3)
    
    # Stop service
    service.stop()
    
    # Check that memory monitor thread is stopped
    assert not service.memory_monitor_thread.is_alive()


def test_batch_size_adjustment():
    """Test batch size adjustment based on memory usage."""
    # Create service
    service = TestService()
    
    # Set initial batch size
    service.current_batch_size = 100
    
    # Test reducing batch size
    service._reduce_batch_size()
    assert service.current_batch_size == 70  # 100 * 0.7
    
    # Test increasing batch size
    service._increase_batch_size()
    assert service.current_batch_size == 84  # 70 * 1.2
    
    # Test minimum batch size
    service.current_batch_size = 5
    service.min_batch_size = 10
    service._reduce_batch_size()
    assert service.current_batch_size == 10  # min_batch_size
    
    # Test maximum batch size
    service.current_batch_size = 900
    service.max_batch_size = 1000
    service._increase_batch_size()
    assert service.current_batch_size == 1000  # max_batch_size