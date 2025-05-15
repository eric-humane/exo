"""
Image generation endpoints for the chat API.

This module contains handlers for image-related endpoints like image generation
and animation creation.
"""

import asyncio
import json
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from aiohttp import web
from PIL import Image

from exo import DEBUG
from exo.helpers import get_exo_images_dir
from exo.apputil import create_animation_mp4
from exo.models import build_base_shard

from .utils import base64_decode, get_progress_bar


class ImageEndpoints:
    """
    Endpoints for image generation in the chat API.
    
    This class handles endpoints for generating images and animations.
    """
    
    def __init__(
        self,
        node,
        inference_engine_classname: str,
        response_timeout: int = 90
    ):
        """
        Initialize the image endpoints.
        
        Args:
            node: Node for processing
            inference_engine_classname: Inference engine class name
            response_timeout: Response timeout in seconds
        """
        self.node = node
        self.inference_engine_classname = inference_engine_classname
        self.response_timeout = response_timeout
        
        # Initialize images directory
        self.images_dir = get_exo_images_dir()
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    async def handle_post_image_generations(self, request: web.Request) -> web.Response:
        """
        Handle POST /v1/image/generations endpoint.
        
        This endpoint generates images based on text prompts.
        
        Args:
            request: Request object
            
        Returns:
            Response with generated images
        """
        data = await request.json()

        if DEBUG >= 2: 
            print(f"Handling image generations request from {request.remote}: {data}")
            
        stream = data.get("stream", False)
        model = data.get("model", "")
        prompt = data.get("prompt", "")
        image_url = data.get("image_url", "")
        
        if DEBUG >= 2: 
            print(f"model: {model}, prompt: {prompt}, stream: {stream}")
            
        shard = build_base_shard(model, self.inference_engine_classname)
        
        if DEBUG >= 2: 
            print(f"shard: {shard}")
            
        if not shard:
            return web.json_response({
                "error": f"Unsupported model: {model} with inference engine {self.inference_engine_classname}"
            }, status=400)

        request_id = str(uuid.uuid4())
        callback_id = f"chatgpt-api-wait-response-{request_id}"
        callback = self.node.on_token.register(callback_id)
        
        try:
            # Process image if provided
            if image_url and image_url is not None:
                img = base64_decode(image_url)
            else:
                img = None
                
            # Start processing the prompt
            await asyncio.wait_for(
                asyncio.shield(
                    asyncio.create_task(
                        self.node.process_prompt(
                            shard, 
                            prompt, 
                            request_id=request_id, 
                            inference_state={"image": img}
                        )
                    )
                ), 
                timeout=self.response_timeout
            )

            # Set up streaming response
            response = web.StreamResponse(
                status=200, 
                reason='OK', 
                headers={
                    'Content-Type': 'application/octet-stream',
                    "Cache-Control": "no-cache",
                }
            )
            await response.prepare(request)

            # Track stream task
            stream_task = None

            # Define result handler
            async def stream_image(_request_id: str, result, is_finished: bool):
                if isinstance(result, list):
                    # Progress update
                    await response.write(
                        json.dumps({
                            'progress': get_progress_bar((result[0]), (result[1]))
                        }).encode('utf-8') + b'\n'
                    )

                elif isinstance(result, np.ndarray):
                    try:
                        # Convert result to image
                        im = Image.fromarray(np.array(result))
                        
                        # Save the image to a file
                        image_filename = f"{_request_id}.png"
                        image_path = self.images_dir / image_filename
                        im.save(image_path)
                        
                        # Get URL for the saved image
                        try:
                            image_url = request.app.router['static_images'].url_for(filename=image_filename)
                            base_url = f"{request.scheme}://{request.host}"
                            full_image_url = base_url + str(image_url)
                            
                            await response.write(
                                json.dumps({
                                    'images': [{
                                        'url': str(full_image_url), 
                                        'content_type': 'image/png'
                                    }]
                                }).encode('utf-8') + b'\n'
                            )
                        except KeyError as e:
                            if DEBUG >= 2: 
                                print(f"Error getting image URL: {e}")
                            # Fallback to direct file path if URL generation fails
                            await response.write(
                                json.dumps({
                                    'images': [{
                                        'url': str(image_path), 
                                        'content_type': 'image/png'
                                    }]
                                }).encode('utf-8') + b'\n'
                            )
                        
                        if is_finished:
                            await response.write_eof()
                        
                    except Exception as e:
                        if DEBUG >= 2: 
                            print(f"Error processing image: {e}")
                            traceback.print_exc()
                        await response.write(
                            json.dumps({'error': str(e)}).encode('utf-8') + b'\n'
                        )

            # Define callback handler
            def on_result(_request_id: str, result, is_finished: bool):
                nonlocal stream_task
                stream_task = asyncio.create_task(
                    stream_image(_request_id, result, is_finished)
                )
                return _request_id == request_id and is_finished

            # Wait for results
            await callback.wait(on_result, timeout=self.response_timeout*10)

            if stream_task:
                # Wait for the stream task to complete before returning
                await stream_task

            return response

        except Exception as e:
            if DEBUG >= 2: 
                traceback.print_exc()
            return web.json_response({
                "detail": f"Error processing prompt (see logs with DEBUG>=2): {str(e)}"
            }, status=500)
    
    async def handle_create_animation(self, request: web.Request) -> web.Response:
        """
        Handle POST /create_animation endpoint.
        
        This endpoint creates an animation from a still image.
        
        Args:
            request: Request object
            
        Returns:
            Response with animation details
        """
        try:
            data = await request.json()
            replacement_image_path = data.get("replacement_image_path")
            device_name = data.get("device_name", "Local Device")
            prompt_text = data.get("prompt", "")

            if DEBUG >= 2: 
                print(f"Creating animation with params: replacement_image={replacement_image_path}, device={device_name}, prompt={prompt_text}")

            if not replacement_image_path:
                return web.json_response({"error": "replacement_image_path is required"}, status=400)

            # Create temp directory if it doesn't exist
            tmp_dir = Path(tempfile.gettempdir()) / "exo_animations"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique output filename in temp directory
            output_filename = f"animation_{uuid.uuid4()}.mp4"
            output_path = str(tmp_dir / output_filename)

            if DEBUG >= 2: 
                print(f"Animation temp directory: {tmp_dir}, output file: {output_path}, directory exists: {tmp_dir.exists()}, directory permissions: {oct(tmp_dir.stat().st_mode)[-3:]}")

            # Create the animation
            create_animation_mp4(replacement_image_path, output_path, device_name, prompt_text)

            return web.json_response({"status": "success", "output_path": output_path})

        except Exception as e:
            if DEBUG >= 2: 
                traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)