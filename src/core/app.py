"""
Core Application

This module contains the main application class for the Trading Algorithm System.
"""

import logging
import signal
import sys
from typing import Any, Dict, List, Optional, Set, Type

from src.config import Config
from src.services.base_service import BaseService
from src.utils.error_handling import AppError
from src.utils.logging_service import setup_logger


class AppError(AppError):
    """Exception raised for application errors."""

    pass


class App:
    """
    Main application class for the Trading Algorithm System.

    This class is responsible for:
    - Application lifecycle management
    - Service coordination
    - Signal handling
    - Configuration management
    """

    def __init__(self):
        """
        Initialize the application.
        """
        self.logger = setup_logger("app")
        self.services: Dict[str, BaseService] = {}
        self.running = False
        self.shutdown_requested = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def register_service(self, service: BaseService) -> None:
        """
        Register a service with the application.

        Args:
            service: The service to register.

        Raises:
            AppError: If a service with the same name is already registered.
        """
        if service.name in self.services:
            raise AppError(f"Service {service.name} is already registered")
        
        self.services[service.name] = service
        self.logger.info(f"Registered service: {service.name}")

    def start(self) -> None:
        """
        Start the application.

        This method starts all registered services.

        Raises:
            AppError: If the application fails to start.
        """
        if self.running:
            self.logger.warning("Application is already running")
            return
        
        try:
            self.logger.info("Starting application")
            
            # Start all services
            for name, service in self.services.items():
                self.logger.info(f"Starting service: {name}")
                service.start()
            
            self.running = True
            self.logger.info("Application started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start application: {str(e)}")
            self.stop()  # Stop any services that were started
            raise AppError(f"Failed to start application: {str(e)}")

    def stop(self) -> None:
        """
        Stop the application.

        This method stops all registered services.
        """
        if not self.running:
            self.logger.warning("Application is not running")
            return
        
        self.logger.info("Stopping application")
        
        # Stop all services in reverse order
        for name, service in reversed(list(self.services.items())):
            try:
                self.logger.info(f"Stopping service: {name}")
                service.stop()
            except Exception as e:
                self.logger.error(f"Error stopping service {name}: {str(e)}")
        
        self.running = False
        self.logger.info("Application stopped")

    def run(self) -> None:
        """
        Run the application until shutdown is requested.
        """
        try:
            self.start()
            
            self.logger.info("Application running, press Ctrl+C to stop")
            
            # Run until shutdown is requested
            while self.running and not self.shutdown_requested:
                signal.pause()
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """
        Handle signals (e.g., SIGINT, SIGTERM).

        Args:
            signum: The signal number.
            frame: The current stack frame.
        """
        if signum == signal.SIGINT:
            self.logger.info("Received SIGINT, shutting down")
        elif signum == signal.SIGTERM:
            self.logger.info("Received SIGTERM, shutting down")
        else:
            self.logger.info(f"Received signal {signum}, shutting down")
        
        self.shutdown_requested = True
        
        # Stop the application if it's running
        if self.running:
            self.stop()

    def get_service(self, name: str) -> Optional[BaseService]:
        """
        Get a registered service by name.

        Args:
            name: The name of the service.

        Returns:
            The service, or None if not found.
        """
        return self.services.get(name)

    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the application.

        Returns:
            A dictionary containing application status information.
        """
        service_statuses = {}
        
        for name, service in self.services.items():
            try:
                service_statuses[name] = service.check_health()
            except Exception as e:
                service_statuses[name] = {"status": "error", "error": str(e)}
        
        return {
            "running": self.running,
            "services": service_statuses,
            "mode": Config.get("APP_MODE", "development"),
        }