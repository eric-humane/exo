#!/usr/bin/env python3
"""
Example demonstrating the AsyncResource pattern for exo.

This example shows how to use the AsyncResource pattern for managing
resources in a distributed exo deployment, including proper initialization,
cleanup, and error handling.
"""

import asyncio
import argparse
import uuid
import sys
import os
import time
from typing import List, Optional, Dict, Any

# Add the parent directory to sys.path to import exo modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exo.utils.async_resources import AsyncResource, ResourceState, AsyncResourceGroup
from exo.utils.resource_monitor import ResourceMonitor, print_resource_states
from exo.networking.discovery import Discovery
from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.server import Server
from exo.inference.inference_engine import InferenceEngine
from exo.inference.dummy_inference_engine import DummyInferenceEngine
from exo.orchestration.node import Node
from exo.download.shard_download import ShardDownloader
from exo.download.new_shard_download import NewShardDownloader
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy


async def run_node(node_id: str, monitor: bool = True) -> None:
    """
    Run an exo node with proper resource management.
    
    Args:
        node_id: ID for the node
        monitor: Whether to run a resource monitor
    """
    print(f"Starting node {node_id}...")
    
    # Create a resource group for this node
    resource_group = AsyncResourceGroup(f"node_{node_id}")
    
    try:
        # Create and add resources to the group
        print("Creating resources...")
        
        # Discovery resource
        discovery = UDPDiscovery(node_id)
        await resource_group.add_resource("discovery", discovery)
        
        # Server
        server = Server(node_id, "localhost", 0)
        await resource_group.add_resource("server", server)
        
        # Inference engine (dummy for this example)
        inference_engine = DummyInferenceEngine()
        await resource_group.add_resource("inference_engine", inference_engine)
        
        # Shard downloader
        shard_downloader = NewShardDownloader()
        await resource_group.add_resource("shard_downloader", shard_downloader)
        
        # Partitioning strategy
        partitioning_strategy = RingMemoryWeightedPartitioningStrategy()
        
        # Create the node
        node = Node(
            node_id,
            server,
            inference_engine,
            discovery,
            shard_downloader,
            partitioning_strategy
        )
        
        # Start a resource monitor if requested
        resource_monitor = None
        if monitor:
            print("Starting resource monitor...")
            resource_monitor = ResourceMonitor(check_interval=30.0)
            await resource_monitor.start()
        
        # Start the node
        print("Starting node...")
        await node.start()
        print(f"Node {node_id} started successfully")
        
        # Print resource states
        print_resource_states()
        
        try:
            # Keep the node running
            while True:
                await asyncio.sleep(10)
                print(f"Node {node_id} still running, connected to {len(node.peers)} peers")
                
                if monitor and resource_monitor:
                    # Print stats from the monitor
                    stats = resource_monitor.get_latest_stats()
                    problem_resources = resource_monitor.get_problem_resources()
                    
                    if problem_resources:
                        print(f"Problem resources: {len(problem_resources)}")
                        for resource_id in problem_resources:
                            print(f"  - {resource_id}")
        except asyncio.CancelledError:
            print("Node shutting down...")
        finally:
            # Stop the node
            print("Stopping node...")
            await node.stop()
            
            # Stop the resource monitor
            if monitor and resource_monitor:
                print("Stopping resource monitor...")
                await resource_monitor.stop()
            
            # Clean up the resource group
            print("Cleaning up resources...")
            await resource_group.cleanup()
            
            print("Node shut down successfully")
    except Exception as e:
        print(f"Error running node: {e}", file=sys.stderr)
        # Make sure to clean up resources even on error
        await resource_group.cleanup()
        raise


def main() -> None:
    """Main entry point for the example."""
    parser = argparse.ArgumentParser(description="Run an exo node with AsyncResource pattern")
    parser.add_argument("--id", help="Node ID (default: random UUID)")
    parser.add_argument("--no-monitor", action="store_true", help="Disable resource monitoring")
    
    args = parser.parse_args()
    node_id = args.id or str(uuid.uuid4())[:8]
    monitor = not args.no_monitor
    
    try:
        asyncio.run(run_node(node_id, monitor))
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()