"""
Command-line interface for managing and monitoring AsyncResource instances.

This module provides utilities for diagnosing and controlling AsyncResource instances
from the command line or interactive Python sessions.
"""

import asyncio
import argparse
import time
import sys
import json
from typing import List, Optional, Dict, Any

from exo.utils.async_resources import AsyncResource, ResourceState
from exo.utils.resource_monitor import get_resource_health_report, print_resource_states, get_resource_stats


async def list_resources() -> None:
    """List all registered resources with their states."""
    print_resource_states()
    
    
async def health_check() -> None:
    """Run a health check on all resources and print the results."""
    report = await get_resource_health_report()
    print(report)
    
    
async def reset_resource(resource_id: str, resource_type: Optional[str] = None) -> None:
    """
    Reset a specific resource by ID.
    
    Args:
        resource_id: ID of the resource to reset
        resource_type: Type of the resource (optional, but helps narrow down the search)
    """
    # Find the resource
    resource = None
    
    if resource_type:
        # If type is provided, look only in that type
        resources = AsyncResource._resource_registry.get(resource_type, [])
        for r in resources:
            if r.id == resource_id:
                resource = r
                break
    else:
        # Otherwise, search all types
        for resources in AsyncResource._resource_registry.values():
            for r in resources:
                if r.id == resource_id:
                    resource = r
                    break
            if resource:
                break
                
    if not resource:
        print(f"Resource not found: {resource_id}")
        return
        
    print(f"Resetting resource: {resource.id} (state: {resource.state.name})")
    
    try:
        # Try to cleanup and reinitialize
        await resource.cleanup()
        print(f"Resource cleaned up, state: {resource.state.name}")
        
        await resource.initialize()
        print(f"Resource reinitialized, state: {resource.state.name}")
        
        # Run a health check
        is_healthy = await resource.check_health()
        print(f"Health check: {'HEALTHY' if is_healthy else 'UNHEALTHY'}")
        
    except Exception as e:
        print(f"Error resetting resource: {str(e)}")
        
        
async def cleanup_resources() -> None:
    """Clean up all registered resources."""
    stats_before = await get_resource_stats()
    
    print("Cleaning up all resources...")
    
    # Clean up by type to avoid modifying during iteration
    for resource_type, resources in list(AsyncResource._resource_registry.items()):
        if not resources:
            continue
            
        print(f"Cleaning up {len(resources)} {resource_type} resources")
        
        # Make a copy of the list to avoid modification during iteration
        for resource in list(resources):
            try:
                await resource.cleanup()
                print(f"  Cleaned up {resource.id}")
            except Exception as e:
                print(f"  Error cleaning up {resource.id}: {str(e)}")
                
    # Print final state
    stats_after = await get_resource_stats()
    print("\nResource counts before cleanup:")
    for resource_type, state_counts in stats_before.items():
        print(f"  {resource_type}: {sum(state_counts.values())} resources")
        
    print("\nResource counts after cleanup:")
    for resource_type, state_counts in stats_after.items():
        print(f"  {resource_type}: {sum(state_counts.values())} resources")
        
        
async def export_resource_data(output_file: Optional[str] = None) -> None:
    """
    Export detailed resource data to JSON.
    
    Args:
        output_file: Path to output file. If None, prints to stdout.
    """
    data = {
        "timestamp": time.time(),
        "resources": {}
    }
    
    # Collect data by resource type
    for resource_type, resources in AsyncResource._resource_registry.items():
        data["resources"][resource_type] = []
        
        for resource in resources:
            resource_data = {
                "id": resource.id,
                "state": resource.state.name,
                "is_initialized": resource.is_initialized,
                "is_usable": resource.is_usable,
                "is_healthy": resource.is_healthy,
                "in_use": resource.in_use
            }
            
            # Add error info if present
            if resource.last_error:
                resource_data["error"] = str(resource.last_error)
                
            # Try to get additional attributes dynamically
            for attr in ["address", "display_name", "_address", "_connection_attempts"]:
                if hasattr(resource, attr):
                    value = getattr(resource, attr)
                    if not callable(value):
                        resource_data[attr.lstrip("_")] = value
                        
            data["resources"][resource_type].append(resource_data)
            
    # Output the data
    json_data = json.dumps(data, indent=2)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(json_data)
        print(f"Resource data exported to {output_file}")
    else:
        print(json_data)


def main() -> None:
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(description="Manage and monitor AsyncResource instances")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all resources")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Run health check on all resources")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset a specific resource")
    reset_parser.add_argument("resource_id", help="ID of the resource to reset")
    reset_parser.add_argument("--type", help="Type of the resource (optional)")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up all resources")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export resource data to JSON")
    export_parser.add_argument("--output", help="Output file path (default: stdout)")
    
    args = parser.parse_args()
    
    # Create and run the appropriate coroutine
    coro = None
    
    if args.command == "list":
        coro = list_resources()
    elif args.command == "health":
        coro = health_check()
    elif args.command == "reset":
        coro = reset_resource(args.resource_id, args.type)
    elif args.command == "cleanup":
        coro = cleanup_resources()
    elif args.command == "export":
        coro = export_resource_data(args.output)
    else:
        parser.print_help()
        return
        
    # Run the coroutine in an event loop
    asyncio.run(coro)


if __name__ == "__main__":
    main()