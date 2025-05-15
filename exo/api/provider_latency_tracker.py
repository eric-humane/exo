"""
Provider latency tracking for LiteLLM integration.

This module provides a latency tracking system for model providers,
helping exo select the fastest client with valid keys for model requests.
"""

import time
import asyncio
import statistics
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, deque
import logging

class ModelProviderLatencyTracker:
    """
    Tracks latency metrics for model providers to enable intelligent routing.
    
    This class maintains:
    1. Recent latency measurements for each provider
    2. Provider health status (based on successful/failed requests)
    3. API key validity state
    4. Statistical aggregations for routing decisions
    """
    
    def __init__(
        self, 
        window_size: int = 20, 
        success_threshold: float = 0.8,
        max_error_age: float = 300.0
    ):
        """
        Initialize the latency tracker.
        
        Args:
            window_size: Number of measurements to keep per provider
            success_threshold: Minimum success rate to consider a provider healthy
            max_error_age: Maximum age in seconds to consider an error relevant
        """
        self._window_size = window_size
        self._success_threshold = success_threshold
        self._max_error_age = max_error_age
        
        # Map of provider -> model -> list of latency measurements
        self._latencies: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=window_size)))
        
        # Map of provider -> set of invalid API keys
        self._invalid_keys: Dict[str, Set[str]] = defaultdict(set)
        
        # Map of provider -> list of recent errors with timestamps
        self._recent_errors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Lock for thread-safety
        self._lock = asyncio.Lock()
    
    async def record_latency(
        self, 
        provider: str, 
        model: str, 
        latency: float, 
        api_key: Optional[str] = None,
        success: bool = True,
        error: Optional[Exception] = None
    ) -> None:
        """
        Record a latency measurement for a provider and model.
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic')
            model: The model name
            latency: The measured latency in seconds
            api_key: API key used (optional, for tracking validity)
            success: Whether the request was successful
            error: Exception if the request failed
        """
        async with self._lock:
            # Record the latency
            if success:
                self._latencies[provider][model].append(latency)
            
            # Handle API key validity
            if not success and api_key:
                if isinstance(error, (ValueError, PermissionError)) and "api key" in str(error).lower():
                    # This looks like an invalid API key error
                    self._invalid_keys[provider].add(api_key)
                    logging.warning(f"Marked API key for {provider} as invalid")
            
            # Record error if present
            if not success and error:
                self._recent_errors[provider].append({
                    "timestamp": time.time(),
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "model": model
                })
                
                # Clean up old errors
                self._clean_old_errors(provider)
    
    def _clean_old_errors(self, provider: str) -> None:
        """
        Remove errors older than max_error_age.
        
        Args:
            provider: The provider to clean errors for
        """
        now = time.time()
        self._recent_errors[provider] = [
            error for error in self._recent_errors[provider]
            if now - error["timestamp"] < self._max_error_age
        ]
    
    async def get_avg_latency(self, provider: str, model: str) -> Optional[float]:
        """
        Get the average latency for a provider and model.
        
        Args:
            provider: The provider name
            model: The model name
            
        Returns:
            Average latency in seconds or None if no data
        """
        async with self._lock:
            latencies = self._latencies.get(provider, {}).get(model, [])
            if not latencies:
                return None
            return statistics.mean(latencies) if latencies else None
    
    async def is_provider_healthy(self, provider: str) -> bool:
        """
        Check if a provider is considered healthy based on recent errors.
        
        Args:
            provider: The provider name
            
        Returns:
            True if the provider is healthy, False otherwise
        """
        async with self._lock:
            # Clean up old errors
            self._clean_old_errors(provider)
            
            # Count recent errors
            error_count = len(self._recent_errors.get(provider, []))
            
            # If we have no data, assume healthy
            if error_count == 0:
                return True
            
            # Check all models for this provider
            total_requests = 0
            for model_latencies in self._latencies.get(provider, {}).values():
                total_requests += len(model_latencies)
            
            # If we have no successful requests but have errors, consider unhealthy
            if total_requests == 0:
                return False
            
            # Calculate success rate
            success_rate = total_requests / (total_requests + error_count)
            return success_rate >= self._success_threshold
    
    async def is_key_valid(self, provider: str, api_key: str) -> bool:
        """
        Check if an API key is known to be valid for a provider.
        
        Args:
            provider: The provider name
            api_key: The API key to check
            
        Returns:
            True if the key is considered valid, False otherwise
        """
        async with self._lock:
            # If the key is in the invalid set, it's invalid
            return api_key not in self._invalid_keys.get(provider, set())
    
    async def get_fastest_provider(self, model_type: str) -> Optional[str]:
        """
        Determine the fastest healthy provider for a model type.
        
        Args:
            model_type: The type of model (e.g., 'gpt-4', 'claude-3')
            
        Returns:
            Name of the fastest provider or None if no healthy providers
        """
        async with self._lock:
            provider_latencies = {}
            
            # Calculate average latency for each provider that has this model type
            for provider, model_latencies in self._latencies.items():
                # Check if this provider has models matching the type
                matching_models = [
                    model for model in model_latencies.keys()
                    if model_type.lower() in model.lower()
                ]
                
                if not matching_models:
                    continue
                
                # Check provider health
                if not await self.is_provider_healthy(provider):
                    continue
                
                # Calculate average latency across all matching models
                all_latencies = []
                for model in matching_models:
                    all_latencies.extend(model_latencies[model])
                
                if all_latencies:
                    provider_latencies[provider] = statistics.mean(all_latencies)
            
            # Return the provider with lowest latency, or None if no data
            if not provider_latencies:
                return None
                
            return min(provider_latencies.items(), key=lambda x: x[1])[0]
    
    async def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all providers.
        
        Returns:
            Dictionary with provider statistics
        """
        async with self._lock:
            stats = {}
            
            for provider, model_latencies in self._latencies.items():
                provider_stats = {
                    "models": {},
                    "error_count": len(self._recent_errors.get(provider, [])),
                    "is_healthy": await self.is_provider_healthy(provider),
                    "invalid_key_count": len(self._invalid_keys.get(provider, set()))
                }
                
                # Calculate per-model statistics
                for model, latencies in model_latencies.items():
                    if latencies:
                        provider_stats["models"][model] = {
                            "avg_latency": statistics.mean(latencies),
                            "min_latency": min(latencies),
                            "max_latency": max(latencies),
                            "sample_count": len(latencies)
                        }
                
                stats[provider] = provider_stats
            
            return stats