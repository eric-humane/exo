"""
Resource monitoring and diagnostics for exo AsyncResources.

This module provides tools to monitor and diagnose issues with AsyncResource instances,
including health reporting, state visualization, and resource statistics.
"""

import asyncio
import time
from typing import Dict, List, Optional, Set, Tuple, Any
import json
from exo.utils.async_resources import AsyncResource, ResourceState


async def get_resource_stats() -> Dict[str, Dict[str, int]]:
    """
    Get statistics about all registered resources in the system.
    
    Returns:
        Dict mapping resource types to counts by state.
    """
    stats = {}
    
    # Collect all registered resources by type through reflection
    for resource_type, resources in AsyncResource._resource_registry.items():
        type_stats = {state.name: 0 for state in ResourceState}
        for resource in resources:
            type_stats[resource.state.name] += 1
        
        stats[resource_type] = type_stats
    
    return stats


async def get_resource_health_report() -> str:
    """
    Generate a detailed health report for all resources.
    
    Returns:
        A formatted string with resource health information.
    """
    report = ["=== AsyncResource Health Report ==="]
    
    # Get stats
    stats = await get_resource_stats()
    total_resources = sum(sum(state_counts.values()) for state_counts in stats.values())
    
    report.append(f"\nTotal Resources: {total_resources}")
    
    # Add stats by type
    for resource_type, state_counts in stats.items():
        report.append(f"\n{resource_type}: {sum(state_counts.values())} instances")
        for state, count in state_counts.items():
            if count > 0:
                report.append(f"  {state}: {count}")
    
    # Add details for problematic resources
    problem_resources = []
    for type_resources in AsyncResource._resource_registry.values():
        for resource in type_resources:
            if resource.state not in (ResourceState.READY, ResourceState.UNINITIALIZED):
                problem_resources.append(resource)
    
    if problem_resources:
        report.append("\nResources with Issues:")
        for resource in problem_resources:
            report.append(f"  - {resource.id} is {resource.state.name}")
            if resource.last_error:
                report.append(f"    Error: {str(resource.last_error)}")
            
            # If resource is initialized, check health and report
            if resource.is_usable:
                try:
                    is_healthy = await asyncio.wait_for(resource.check_health(), timeout=5.0)
                    report.append(f"    Current health check: {'HEALTHY' if is_healthy else 'UNHEALTHY'}")
                except asyncio.TimeoutError:
                    report.append(f"    Current health check: TIMEOUT")
                except Exception as e:
                    report.append(f"    Current health check failed: {str(e)}")
    
    report.append("\n==================================")
    return "\n".join(report)


def print_resource_states() -> None:
    """Print a summary of all resource states to stdout."""
    print("\n=== AsyncResource States ===")
    for resource_type, resources in AsyncResource._resource_registry.items():
        print(f"\n{resource_type}: {len(resources)} instances")
        
        # Group by state
        state_groups = {}
        for resource in resources:
            state = resource.state.name
            if state not in state_groups:
                state_groups[state] = []
            state_groups[state].append(resource)
        
        # Print counts by state
        for state in ResourceState:
            count = len(state_groups.get(state.name, []))
            if count > 0:
                print(f"  {state.name}: {count}")
        
        # Print details for non-READY states
        problematic_states = [
            ResourceState.ERROR, 
            ResourceState.DEGRADED, 
            ResourceState.INITIALIZING, 
            ResourceState.CLOSING
        ]
        
        for state in problematic_states:
            resources_in_state = state_groups.get(state.name, [])
            if resources_in_state:
                print(f"\n  {state.name} resources:")
                for resource in resources_in_state:
                    error_msg = f" - Error: {resource.last_error}" if resource.last_error else ""
                    print(f"    - {resource.id}{error_msg}")
    
    print("\n=============================")


async def get_resource_graph() -> Dict[str, Any]:
    """
    Generate a graph representation of resource relationships.
    
    Returns:
        A dictionary with nodes and edges representing resources and their relationships.
    """
    graph = {
        "nodes": [],
        "edges": []
    }
    
    # First add all resources as nodes
    for resource_type, resources in AsyncResource._resource_registry.items():
        for resource in resources:
            node = {
                "id": resource.id,
                "type": resource_type,
                "state": resource.state.name,
                "healthy": resource.is_healthy,
                "usable": resource.is_usable
            }
            
            # Add error info if present
            if resource.last_error:
                node["error"] = str(resource.last_error)
                
            graph["nodes"].append(node)
    
    # For future: Add edges based on resource dependencies if we track those
    
    return graph


class ResourceMonitor:
    """
    Monitor for AsyncResources that periodically checks health and collects statistics.
    
    Usage:
    ```python
    # Create and start the monitor
    monitor = ResourceMonitor()
    await monitor.start()
    
    # Get the latest stats
    stats = monitor.get_latest_stats()
    
    # Stop the monitor when done
    await monitor.stop()
    ```
    """
    
    def __init__(self, check_interval: float = 60.0):
        """
        Initialize a resource monitor.
        
        Args:
            check_interval: How often to check resource health, in seconds.
        """
        self._check_interval = check_interval
        self._monitor_task = None
        self._latest_stats = {}
        self._last_check_time = 0.0
        self._is_running = False
        self._problem_resources: Set[str] = set()
        
    async def start(self) -> None:
        """Start the resource monitor."""
        if self._is_running:
            return
            
        self._is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
    async def stop(self) -> None:
        """Stop the resource monitor."""
        if not self._is_running:
            return
            
        self._is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            
    def get_latest_stats(self) -> Dict[str, Dict[str, int]]:
        """Get the latest resource statistics."""
        return self._latest_stats
    
    def get_problem_resources(self) -> Set[str]:
        """Get the IDs of resources that are having problems."""
        return self._problem_resources.copy()
    
    async def get_detailed_report(self) -> str:
        """Get a detailed report of all resources."""
        return await get_resource_health_report()
        
    async def _monitor_loop(self) -> None:
        """Background task that periodically checks resources."""
        try:
            while self._is_running:
                await self._check_resources()
                await asyncio.sleep(self._check_interval)
        except asyncio.CancelledError:
            # Normal cancellation
            return
            
    async def _check_resources(self) -> None:
        """Check all resources and update statistics."""
        try:
            self._last_check_time = time.time()
            
            # Get fresh statistics
            self._latest_stats = await get_resource_stats()
            
            # Clear previous problem resources
            self._problem_resources.clear()
            
            # Check for problematic resources
            for resource_type, resources in AsyncResource._resource_registry.items():
                for resource in resources:
                    if resource.state not in (ResourceState.READY, ResourceState.UNINITIALIZED):
                        self._problem_resources.add(resource.id)
                        
                        # If it's degraded, try a health check
                        if resource.state == ResourceState.DEGRADED:
                            try:
                                await asyncio.wait_for(resource.check_health(), timeout=5.0)
                            except Exception:
                                # Just ignore errors; we're already tracking it as problematic
                                pass
        except Exception:
            # Ignore errors in the monitoring loop
            pass