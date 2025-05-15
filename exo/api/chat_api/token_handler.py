"""
Token handling module for the chat API.

This module handles token streaming, processing, and queuing for chat completions.
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Optional
import traceback

from exo import DEBUG


class TokenHandler:
    """
    Handles token processing and streaming for chat API.
    
    This class manages token queues, processes incoming tokens,
    and provides methods for handling token streaming.
    """
    
    def __init__(self):
        """Initialize the token handler."""
        # Dictionary mapping request_id to token queues
        self.token_queues: Dict[str, asyncio.Queue] = {}
        
        # Dictionary mapping request_id to active streaming tasks
        self.stream_tasks: Dict[str, asyncio.Task] = {}
    
    def create_queue(self, request_id: str) -> asyncio.Queue:
        """
        Create a token queue for a request.
        
        Args:
            request_id: Unique identifier for the request
            
        Returns:
            The created queue
        """
        self.token_queues[request_id] = asyncio.Queue()
        return self.token_queues[request_id]
    
    def get_queue(self, request_id: str) -> Optional[asyncio.Queue]:
        """
        Get the token queue for a request if it exists.
        
        Args:
            request_id: Unique identifier for the request
            
        Returns:
            The queue or None if it doesn't exist
        """
        return self.token_queues.get(request_id)
    
    def remove_queue(self, request_id: str) -> None:
        """
        Remove a token queue.
        
        Args:
            request_id: Unique identifier for the request
        """
        if request_id in self.token_queues:
            del self.token_queues[request_id]
    
    def track_task(self, request_id: str, task: asyncio.Task) -> None:
        """
        Track a streaming task.
        
        Args:
            request_id: Unique identifier for the request
            task: Task to track
        """
        self.stream_tasks[request_id] = task
    
    def remove_task(self, request_id: str) -> None:
        """
        Remove a tracked task.
        
        Args:
            request_id: Unique identifier for the request
        """
        if request_id in self.stream_tasks:
            del self.stream_tasks[request_id]
    
    def cancel_tasks(self) -> None:
        """Cancel all tracked tasks."""
        for request_id, task in self.stream_tasks.items():
            if not task.done():
                task.cancel()
        self.stream_tasks.clear()
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        self.cancel_tasks()
        self.token_queues.clear()
    
    async def handle_tokens(self, request_id: str, tokens: List[int], is_finished: bool) -> None:
        """
        Process newly generated tokens for a specific request.
        
        This method is called when new tokens are generated. It puts the tokens
        into the appropriate queue for the request.
        
        Args:
            request_id: Unique identifier for the request
            tokens: List of token IDs to process
            is_finished: Flag indicating if this is the final set of tokens
            
        Returns:
            None
        """
        try:
            # Validate inputs
            if not request_id:
                if DEBUG >= 1:
                    logging.warning("Empty request_id provided to handle_tokens")
                return

            if not isinstance(tokens, list):
                if DEBUG >= 1:
                    logging.warning(f"Invalid tokens format for request {request_id}: {type(tokens)}")
                tokens = list(tokens) if hasattr(tokens, '__iter__') else []

            # Safely check if the queue exists for this request
            if request_id not in self.token_queues:
                if DEBUG >= 1:
                    logging.warning(f"Token queue for request {request_id} no longer exists")
                return

            # Put the tokens in the queue
            await self.token_queues[request_id].put((tokens, is_finished))

            if is_finished and DEBUG >= 2:
                logging.info(f"Request {request_id} finished, {len(tokens)} final tokens queued")

        except asyncio.QueueFull:
            if DEBUG >= 1:
                logging.error(f"Token queue for request {request_id} is full")
        except Exception as e:
            if DEBUG >= 1:
                logging.error(f"Error in handle_tokens for request {request_id}: {e}")
                if DEBUG >= 2:
                    traceback.print_exc()
    
    async def wait_for_tokens(
        self, 
        request_id: str, 
        timeout: float = 15.0
    ) -> Tuple[List[int], bool]:
        """
        Wait for tokens from the queue with timeout.
        
        Args:
            request_id: Unique identifier for the request
            timeout: Maximum time to wait in seconds
            
        Returns:
            Tuple of (tokens, is_finished)
            
        Raises:
            asyncio.TimeoutError: If the timeout is reached
            KeyError: If the queue doesn't exist
        """
        if request_id not in self.token_queues:
            raise KeyError(f"No token queue for request {request_id}")
        
        return await asyncio.wait_for(
            self.token_queues[request_id].get(), 
            timeout=timeout
        )