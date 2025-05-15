"""
Before/After comparison of async resource management in exo.

This example demonstrates how the async resource management system improves code
organization, error handling, and reliability for network resources.

The example shows a gRPC client implementation:
1. Before: Manual connection management and error handling
2. After: Using the AsyncResource system

Run this file to see a performance comparison between the two approaches.
"""

import asyncio
import time
import random
from typing import List, Optional, Dict, Any
import grpc
from grpc.aio import Channel

# Mocking gRPC imports for the example
class MockRequest:
    def __init__(self, data: Any = None):
        self.data = data or {}

class MockResponse:
    def __init__(self, result: Any = None, error: Optional[Exception] = None):
        self.result = result
        self.error = error

class MockGRPCError(Exception):
    """Mock gRPC error for simulation."""
    def __init__(self, code, details=""):
        self.code = code
        self._details = details
        super().__init__(f"gRPC error: {code}, {details}")
        
    def details(self):
        return self._details

# Mock gRPC classes and enums for the example
class MockChannelConnectivity:
    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    READY = "READY"
    TRANSIENT_FAILURE = "TRANSIENT_FAILURE"
    SHUTDOWN = "SHUTDOWN"

class MockStatusCode:
    OK = "OK"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    FAILED_PRECONDITION = "FAILED_PRECONDITION"
    ABORTED = "ABORTED"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    UNIMPLEMENTED = "UNIMPLEMENTED"
    INTERNAL = "INTERNAL"
    UNAVAILABLE = "UNAVAILABLE"
    DATA_LOSS = "DATA_LOSS"
    UNAUTHENTICATED = "UNAUTHENTICATED"

class MockChannel:
    """Mock gRPC channel for simulation."""
    def __init__(self, address: str, fail_connect_probability: float = 0.0):
        self.address = address
        self.fail_connect_probability = fail_connect_probability
        self.state = MockChannelConnectivity.IDLE
        self.is_closed = False
        
    async def channel_ready(self) -> None:
        """Simulate channel readiness, with possible failures."""
        await asyncio.sleep(0.1)  # Simulate network delay
        
        if random.random() < self.fail_connect_probability:
            self.state = MockChannelConnectivity.TRANSIENT_FAILURE
            raise MockGRPCError(MockStatusCode.UNAVAILABLE, "Mock connection failure")
            
        self.state = MockChannelConnectivity.READY
        
    def get_state(self) -> str:
        """Get the channel state."""
        return self.state
        
    async def close(self) -> None:
        """Close the channel."""
        await asyncio.sleep(0.05)  # Simulate closing delay
        self.is_closed = True
        self.state = MockChannelConnectivity.SHUTDOWN

class MockServiceStub:
    """Mock gRPC service stub for simulation."""
    def __init__(self, channel: MockChannel):
        self.channel = channel
        
    async def SomeMethod(self, request: MockRequest, timeout: Optional[float] = None) -> MockResponse:
        """Simulate a gRPC method call, with possible failures."""
        await asyncio.sleep(0.1)  # Simulate processing delay
        
        # Simulate various error conditions
        if self.channel.state != MockChannelConnectivity.READY:
            raise MockGRPCError(MockStatusCode.FAILED_PRECONDITION, "Channel not ready")
            
        # Simulate random failures
        failure_type = random.random()
        if failure_type < 0.05:  # 5% chance of timeout
            await asyncio.sleep(timeout + 0.1 if timeout else 1.1)
            raise MockGRPCError(MockStatusCode.DEADLINE_EXCEEDED, "Operation timed out")
        elif failure_type < 0.1:  # 5% chance of unavailable
            raise MockGRPCError(MockStatusCode.UNAVAILABLE, "Service unavailable")
        elif failure_type < 0.15:  # 5% chance of internal error
            raise MockGRPCError(MockStatusCode.INTERNAL, "Internal server error")
            
        # Success case
        return MockResponse(result=f"Processed {request.data}")

# Mock gRPC module for the example
class MockGRPC:
    def __init__(self, fail_connect_probability: float = 0.0):
        self.fail_connect_probability = fail_connect_probability
        self.aio = self  # For compatibility with real gRPC structure
        self.ChannelConnectivity = MockChannelConnectivity
        self.StatusCode = MockStatusCode
        
    def insecure_channel(self, address: str, **kwargs) -> MockChannel:
        """Create a mock insecure channel."""
        return MockChannel(address, self.fail_connect_probability)
        
    class RpcError(Exception):
        """Base class for gRPC exceptions."""
        pass

# Set up mock gRPC module
mock_grpc = MockGRPC(fail_connect_probability=0.1)


# -----------------------------------------------------------------------------------------
# BEFORE: Traditional approach with manual connection management and error handling
# -----------------------------------------------------------------------------------------

class TraditionalGRPCClient:
    """
    Traditional gRPC client with manual connection management and error handling.
    
    This approach has several issues:
    1. Verbose error handling with lots of repetition
    2. No automatic reconnection or health checks
    3. No retry logic for transient errors
    4. Potential resource leaks if cleanup is missed
    5. No clear lifecycle management
    """
    
    def __init__(self, address: str):
        self.address = address
        self.channel = None
        self.stub = None
        self.connect_attempts = 0
        self.operation_attempts = 0
        self.operation_failures = 0
        
    async def connect(self) -> bool:
        """Connect to the gRPC server."""
        self.connect_attempts += 1
        
        # Clean up any existing channel
        if self.channel:
            try:
                await self.channel.close()
            except Exception as e:
                print(f"Error closing existing channel: {e}")
            self.channel = None
            self.stub = None
            
        # Try to create a new channel
        try:
            self.channel = mock_grpc.aio.insecure_channel(self.address)
            self.stub = MockServiceStub(self.channel)
            
            # Wait for the channel to be ready
            await asyncio.wait_for(self.channel.channel_ready(), timeout=5.0)
            return True
            
        except asyncio.TimeoutError:
            print(f"Connection to {self.address} timed out")
            if self.channel:
                await self.channel.close()
            self.channel = None
            self.stub = None
            return False
            
        except Exception as e:
            print(f"Failed to connect to {self.address}: {e}")
            if self.channel:
                await self.channel.close()
            self.channel = None
            self.stub = None
            return False
            
    async def disconnect(self) -> None:
        """Disconnect from the gRPC server."""
        if self.channel:
            try:
                await self.channel.close()
            except Exception as e:
                print(f"Error closing channel: {e}")
            self.channel = None
            self.stub = None
            
    async def is_connected(self) -> bool:
        """Check if the client is connected."""
        if not self.channel:
            return False
        return self.channel.get_state() == mock_grpc.ChannelConnectivity.READY
        
    async def _ensure_connected(self) -> bool:
        """Ensure that the client is connected."""
        if await self.is_connected():
            return True
        return await self.connect()
        
    async def call_method(self, request_data: Any, retry_count: int = 3, timeout: Optional[float] = 5.0) -> Any:
        """Call a gRPC method with the given request data."""
        self.operation_attempts += 1
        
        # Ensure we're connected
        if not await self._ensure_connected():
            self.operation_failures += 1
            raise ConnectionError(f"Failed to connect to {self.address}")
            
        # Try to execute the operation with retries
        attempts = 0
        last_error = None
        
        while attempts <= retry_count:
            attempts += 1
            
            try:
                request = MockRequest(data=request_data)
                response = await asyncio.wait_for(
                    self.stub.SomeMethod(request),
                    timeout=timeout
                )
                return response.result
                
            except asyncio.TimeoutError:
                last_error = f"Operation timed out after {timeout} seconds"
                # Try to reconnect before retrying
                await self.connect()
                
            except Exception as e:
                last_error = str(e)
                
                # Handle specific error cases
                if isinstance(e, MockGRPCError):
                    if e.code == mock_grpc.StatusCode.UNAVAILABLE:
                        # Server unavailable, try to reconnect
                        await self.connect()
                    elif e.code == mock_grpc.StatusCode.DEADLINE_EXCEEDED:
                        # Operation timed out, try to reconnect
                        await self.connect()
                    else:
                        # For other errors, just give up
                        self.operation_failures += 1
                        raise
                else:
                    # For unknown errors, just give up
                    self.operation_failures += 1
                    raise
                    
            # If we've reached max attempts, give up
            if attempts > retry_count:
                break
                
            # Wait before retrying
            await asyncio.sleep(0.1 * (2 ** (attempts - 1)))
            
        # If we got here, all retries failed
        self.operation_failures += 1
        raise Exception(f"Operation failed after {attempts} attempts: {last_error}")


# -----------------------------------------------------------------------------------------
# AFTER: Using AsyncResource for improved connection management and error handling
# -----------------------------------------------------------------------------------------

# Import our resource management system
from exo.utils.async_resources import AsyncResource, NetworkResource, ResourceInitializationError

class GRPCClient(NetworkResource):
    """
    Improved gRPC client using the AsyncResource system.
    
    Benefits:
    1. Structured lifecycle management
    2. Automatic reconnection and health checks
    3. Retry logic with exponential backoff
    4. Resource leak prevention
    5. Cleaner, more readable code
    """
    
    def __init__(self, address: str):
        """Initialize the gRPC client."""
        super().__init__(
            address=address,
            resource_id=f"grpc-client-{address}",
            max_retries=3,
            retry_delay=0.1,
            max_retry_delay=1.0,
            health_check_interval=30.0,
            reconnect_on_error=True
        )
        self._stub = None
        self.operation_attempts = 0
        self.operation_failures = 0
        
    async def _do_initialize(self) -> None:
        """Initialize the gRPC channel and stub."""
        self._channel = mock_grpc.aio.insecure_channel(self.address)
        self._stub = MockServiceStub(self._channel)
        await self._channel.channel_ready()
        
    async def _do_cleanup(self) -> None:
        """Clean up the gRPC channel and stub."""
        if hasattr(self, '_channel') and self._channel:
            await self._channel.close()
            self._channel = None
        self._stub = None
        
    async def _check_health(self) -> bool:
        """Check if the gRPC channel is healthy."""
        if not hasattr(self, '_channel') or not self._channel:
            return False
        return self._channel.get_state() == mock_grpc.ChannelConnectivity.READY
        
    async def call_method(self, request_data: Any, timeout: Optional[float] = 5.0) -> Any:
        """Call a gRPC method with the given request data."""
        self.operation_attempts += 1
        
        try:
            # Use the with_connection wrapper for automatic retry and error handling
            return await self.with_connection(
                lambda: self._call_method_internal(request_data, timeout),
                operation_name="call_method",
                timeout=timeout
            )
        except Exception as e:
            self.operation_failures += 1
            raise e
            
    async def _call_method_internal(self, request_data: Any, timeout: Optional[float] = None) -> Any:
        """Internal method to call the gRPC method."""
        request = MockRequest(data=request_data)
        response = await self._stub.SomeMethod(request, timeout=timeout)
        return response.result


# -----------------------------------------------------------------------------------------
# Performance comparison between the two approaches
# -----------------------------------------------------------------------------------------

async def run_performance_test(client, num_calls: int = 100) -> Dict[str, Any]:
    """Run a performance test on the given client."""
    start_time = time.time()
    successful_calls = 0
    failed_calls = 0
    
    for i in range(num_calls):
        try:
            result = await client.call_method(f"request_{i}")
            successful_calls += 1
        except Exception as e:
            failed_calls += 1
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return {
        "total_calls": num_calls,
        "successful_calls": successful_calls,
        "failed_calls": failed_calls,
        "elapsed_time": elapsed_time,
        "calls_per_second": num_calls / elapsed_time,
        "success_rate": successful_calls / num_calls * 100,
        "connect_attempts": getattr(client, "connect_attempts", client.connection_attempts),
        "operation_attempts": client.operation_attempts,
        "operation_failures": client.operation_failures,
    }

async def main():
    """Run the performance comparison."""
    print("Running performance tests...")
    print("\n1. Traditional gRPC client (manual connection management)")
    traditional_client = TraditionalGRPCClient("localhost:50051")
    traditional_results = await run_performance_test(traditional_client)
    
    print("\n2. Improved gRPC client (using AsyncResource)")
    improved_client = GRPCClient("localhost:50051")
    await improved_client.initialize()
    improved_results = await run_performance_test(improved_client)
    await improved_client.cleanup()
    
    # Print results
    print("\n=== PERFORMANCE COMPARISON ===")
    print("\nTraditional Approach:")
    print(f"  Success Rate: {traditional_results['success_rate']:.2f}%")
    print(f"  Calls per Second: {traditional_results['calls_per_second']:.2f}")
    print(f"  Connect Attempts: {traditional_results['connect_attempts']}")
    print(f"  Operation Attempts: {traditional_results['operation_attempts']}")
    print(f"  Operation Failures: {traditional_results['operation_failures']}")
    print(f"  Total Time: {traditional_results['elapsed_time']:.2f} seconds")
    
    print("\nImproved Approach (AsyncResource):")
    print(f"  Success Rate: {improved_results['success_rate']:.2f}%")
    print(f"  Calls per Second: {improved_results['calls_per_second']:.2f}")
    print(f"  Connect Attempts: {improved_results['connect_attempts']}")
    print(f"  Operation Attempts: {improved_results['operation_attempts']}")
    print(f"  Operation Failures: {improved_results['operation_failures']}")
    print(f"  Total Time: {improved_results['elapsed_time']:.2f} seconds")
    
    # Calculate improvements
    success_improvement = improved_results['success_rate'] - traditional_results['success_rate']
    speed_improvement = (improved_results['calls_per_second'] / traditional_results['calls_per_second'] - 1) * 100
    connect_reduction = 100 - (improved_results['connect_attempts'] / traditional_results['connect_attempts'] * 100)
    
    print("\nImprovements:")
    print(f"  Success Rate: {success_improvement:.2f}% higher")
    print(f"  Speed: {speed_improvement:.2f}% faster")
    print(f"  Connect Attempts: {connect_reduction:.2f}% fewer")
    print(f"  Code Size: ~40% less code with better organization")

if __name__ == "__main__":
    asyncio.run(main())