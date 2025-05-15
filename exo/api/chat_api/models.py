"""
Data models for the chat API.

This module contains the data models used for chat API requests and responses.
"""

from typing import List, Dict, Union, Optional, Literal, Any
import time

class Message:
    """
    Represents a single message in a chat conversation.
    
    Attributes:
        role: The role of the message sender (user, assistant, system)
        content: The message content
        tools: Optional list of tools for function calling
    """
    def __init__(self, role: str, content: Union[str, List[Dict[str, Union[str, Dict[str, str]]]]], 
                 tools: Optional[List[Dict]] = None):
        self.role = role
        self.content = content
        self.tools = tools

    def to_dict(self):
        """Convert the message to a dictionary representation."""
        data = {"role": self.role, "content": self.content}
        if self.tools:
            data["tools"] = self.tools
        return data


class ChatCompletionRequest:
    """
    Represents a request for chat completion.
    
    Attributes:
        model: The model to use for completion
        messages: List of messages in the conversation
        temperature: Sampling temperature
        tools: Optional list of tools for function calling
    """
    def __init__(self, model: str, messages: List[Message], temperature: float, 
                 tools: Optional[List[Dict]] = None):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.tools = tools

    def to_dict(self):
        """Convert the request to a dictionary representation."""
        return {
            "model": self.model, 
            "messages": [message.to_dict() for message in self.messages], 
            "temperature": self.temperature, 
            "tools": self.tools
        }


class PromptSession:
    """
    Represents a saved prompt session.
    
    Attributes:
        request_id: Unique identifier for the request
        timestamp: Time when the session was created
        prompt: The prompt text
    """
    def __init__(self, request_id: str, timestamp: int, prompt: str):
        self.request_id = request_id
        self.timestamp = timestamp
        self.prompt = prompt


def generate_completion(
    chat_request: ChatCompletionRequest,
    tokenizer,
    prompt: str,
    request_id: str,
    tokens: List[int],
    stream: bool,
    finish_reason: Union[Literal["length", "stop"], None],
    object_type: Literal["chat.completion", "text_completion"],
) -> dict:
    """
    Generate a completion response.
    
    Args:
        chat_request: The chat completion request
        tokenizer: Tokenizer to use for decoding
        prompt: Original prompt text
        request_id: Unique identifier for the request
        tokens: List of token IDs generated
        stream: Whether this is a streaming response
        finish_reason: Reason for finishing generation
        object_type: Type of completion object
        
    Returns:
        Dictionary containing the completion response
    """
    completion = {
        "id": f"chatcmpl-{request_id}",
        "object": object_type,
        "created": int(time.time()),
        "model": chat_request.model,
        "system_fingerprint": f"exo_VERSION",  # VERSION will be imported in the main module
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": tokenizer.decode(tokens)},
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
    }

    if not stream:
        completion["usage"] = {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokens),
            "total_tokens": len(tokenizer.encode(prompt)) + len(tokens),
        }

    choice = completion["choices"][0]
    if object_type.startswith("chat.completion"):
        key_name = "delta" if stream else "message"
        choice[key_name] = {"role": "assistant", "content": tokenizer.decode(tokens)}
    elif object_type == "text_completion":
        choice["text"] = tokenizer.decode(tokens)
    else:
        ValueError(f"Unsupported response type: {object_type}")

    return completion


def remap_messages(messages: List[Message]) -> List[Message]:
    """
    Remap messages to handle images.
    
    This function processes the messages to handle images appropriately,
    particularly for multimodal models.
    
    Args:
        messages: List of messages to process
        
    Returns:
        Processed list of messages
    """
    remapped_messages = []
    last_image = None
    for message in messages:
        if not isinstance(message.content, list):
            remapped_messages.append(message)
            continue

        remapped_content = []
        for content in message.content:
            if isinstance(content, dict):
                if content.get("type") in ["image_url", "image"]:
                    image_url = content.get("image_url", {}).get("url") or content.get("image")
                    if image_url:
                        last_image = {"type": "image", "image": image_url}
                        remapped_content.append({"type": "text", "text": "[An image was uploaded but is not displayed here]"})
                else:
                    remapped_content.append(content)
            else:
                remapped_content.append(content)
        remapped_messages.append(Message(role=message.role, content=remapped_content))

    if last_image:
        # Replace the last image placeholder with the actual image content
        for message in reversed(remapped_messages):
            for i, content in enumerate(message.content):
                if isinstance(content, dict):
                    if content.get("type") == "text" and content.get("text") == "[An image was uploaded but is not displayed here]":
                        message.content[i] = last_image
                        return remapped_messages

    return remapped_messages


def build_prompt(tokenizer, _messages: List[Message], tools: Optional[List[Dict]] = None):
    """
    Build a prompt from messages.
    
    Args:
        tokenizer: Tokenizer to use
        _messages: List of messages to build the prompt from
        tools: Optional list of tools for function calling
        
    Returns:
        The constructed prompt
    """
    messages = remap_messages(_messages)
    chat_template_args = {"conversation": [m.to_dict() for m in messages], "tokenize": False, "add_generation_prompt": True}
    if tools: 
        chat_template_args["tools"] = tools

    try:
        prompt = tokenizer.apply_chat_template(**chat_template_args)
        return prompt
    except UnicodeEncodeError:
        # Handle Unicode encoding by ensuring everything is UTF-8
        chat_template_args["conversation"] = [
            {k: v.encode('utf-8').decode('utf-8') if isinstance(v, str) else v 
             for k, v in m.to_dict().items()}
            for m in messages
        ]
        prompt = tokenizer.apply_chat_template(**chat_template_args)
        return prompt


def parse_message(data: dict):
    """
    Parse a message from a dictionary.
    
    Args:
        data: Dictionary containing message data
        
    Returns:
        Message object
        
    Raises:
        ValueError: If the message data is invalid
    """
    if "role" not in data or "content" not in data:
        raise ValueError(f"Invalid message: {data}. Must have 'role' and 'content'")
    return Message(data["role"], data["content"], data.get("tools"))


def parse_chat_request(data: dict, default_model: str):
    """
    Parse a chat completion request from a dictionary.
    
    Args:
        data: Dictionary containing request data
        default_model: Default model to use if not specified
        
    Returns:
        ChatCompletionRequest object
    """
    return ChatCompletionRequest(
        data.get("model", default_model),
        [parse_message(msg) for msg in data["messages"]],
        data.get("temperature", 0.0),
        data.get("tools", None),
    )


class ChatChoice:
    """
    Represents a single choice in a chat completion response.
    
    Attributes:
        index: The index of the choice
        message: The message content
        finish_reason: The reason generation finished
    """
    def __init__(self, index: int, message: Dict[str, str], finish_reason: Optional[str] = None):
        self.index = index
        self.message = message
        self.finish_reason = finish_reason
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "index": self.index,
            "message": self.message,
            "finish_reason": self.finish_reason
        }


class ChatCompletionResponse:
    """
    Represents a response from a chat completion request.
    
    Attributes:
        id: The unique ID of the response
        object: The object type
        created: The timestamp when the response was created
        model: The model used for completion
        choices: The list of completion choices
        usage: Token usage statistics
    """
    def __init__(
        self, 
        id: str, 
        created: int, 
        model: str, 
        choices: List[ChatChoice],
        usage: Optional[Dict[str, int]] = None
    ):
        self.id = id
        self.object = "chat.completion"
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
        self.system_fingerprint = "exo_VERSION"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        response = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "choices": [choice.to_dict() for choice in self.choices]
        }
        if self.usage:
            response["usage"] = self.usage
        return response


class DeleteResponse:
    """
    Represents a response from a delete operation.
    
    Attributes:
        id: The unique ID of the deleted resource
        object: The object type
        deleted: Whether the resource was deleted
    """
    def __init__(self, id: str, object_type: str, deleted: bool = True):
        self.id = id
        self.object = object_type
        self.deleted = deleted
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "object": self.object,
            "deleted": self.deleted
        }


class Model:
    """
    Represents a model in the API.
    
    Attributes:
        id: The unique ID of the model
        object: The object type
        created: The timestamp when the model was created
        owned_by: The owner of the model
    """
    def __init__(self, id: str, created: int, owned_by: str = "exo"):
        self.id = id
        self.object = "model"
        self.created = created
        self.owned_by = owned_by
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by
        }