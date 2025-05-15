"""
Peer authentication system for exo distributed computing.

This module provides mechanisms for authenticating and authorizing peers
in the distributed network to prevent unauthorized access and enhance security.
"""

import os
import asyncio
import time
import base64
import hashlib
import hmac
import json
import uuid
from typing import Dict, Optional, Any, Tuple, List, Set, Callable
from pathlib import Path
import tempfile
from exo.utils import logging


class AuthToken:
    """Represents an authentication token with expiration."""
    
    def __init__(self, token_id: str, token_secret: str, expires_at: float, metadata: Dict[str, Any] = None):
        self.token_id = token_id
        self.token_secret = token_secret
        self.expires_at = expires_at
        self.metadata = metadata or {}
    
    def is_expired(self, current_time: float = None) -> bool:
        """Check if the token has expired."""
        if current_time is None:
            current_time = time.time()
        return current_time > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert token to dictionary for serialization."""
        return {
            "token_id": self.token_id,
            "token_secret": self.token_secret,
            "expires_at": self.expires_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuthToken':
        """Create token from dictionary."""
        return cls(
            token_id=data["token_id"],
            token_secret=data["token_secret"],
            expires_at=data["expires_at"],
            metadata=data.get("metadata", {})
        )


class AuthChallenge:
    """Challenge data for peer authentication."""
    
    def __init__(self, challenge_id: str, challenge_data: str, node_id: str, created_at: float, expires_at: float):
        self.challenge_id = challenge_id
        self.challenge_data = challenge_data
        self.node_id = node_id
        self.created_at = created_at
        self.expires_at = expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert challenge to dictionary for serialization."""
        return {
            "challenge_id": self.challenge_id,
            "challenge_data": self.challenge_data,
            "node_id": self.node_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuthChallenge':
        """Create challenge from dictionary."""
        return cls(
            challenge_id=data["challenge_id"],
            challenge_data=data["challenge_data"],
            node_id=data["node_id"],
            created_at=data["created_at"],
            expires_at=data["expires_at"]
        )
    
    def is_expired(self, current_time: float = None) -> bool:
        """Check if the challenge has expired."""
        if current_time is None:
            current_time = time.time()
        return current_time > self.expires_at


class AuthResponse:
    """Response to an authentication challenge."""
    
    def __init__(self, challenge_id: str, node_id: str, response_data: str, token_id: str):
        self.challenge_id = challenge_id
        self.node_id = node_id
        self.response_data = response_data
        self.token_id = token_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for serialization."""
        return {
            "challenge_id": self.challenge_id,
            "node_id": self.node_id,
            "response_data": self.response_data,
            "token_id": self.token_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuthResponse':
        """Create response from dictionary."""
        return cls(
            challenge_id=data["challenge_id"],
            node_id=data["node_id"],
            response_data=data["response_data"],
            token_id=data["token_id"]
        )


class PeerAuthentication:
    """
    Handles peer authentication in the exo distributed system.
    
    This class provides:
    1. Token generation and validation
    2. Challenge-response authentication
    3. Token storage and management
    """
    
    def __init__(self, node_id: str, tokens_path: Optional[Path] = None, auth_enabled: bool = True):
        self.node_id = node_id
        self.auth_enabled = auth_enabled
        self.tokens_path = tokens_path or Path(tempfile.gettempdir()) / "exo_auth_tokens.json"
        
        # Authentication data
        self.tokens: Dict[str, AuthToken] = {}  # By token_id
        self.challenges: Dict[str, AuthChallenge] = {}  # By challenge_id
        self.authorized_peers: Set[str] = set()  # Set of authorized peer node_ids
        
        # Authentication configuration
        self.token_expiry = 24 * 60 * 60  # 24 hours by default
        self.challenge_expiry = 60  # 60 seconds by default
        
        # Node secret (used for HMAC signing)
        self._node_secret = self._get_or_create_node_secret()
        
        # Periodic cleanup task
        self._cleanup_task = None
        self._shutting_down = False
    
    def _get_or_create_node_secret(self) -> str:
        """
        Get or create a persistent node secret for authentication.
        
        The node secret is used to sign challenges and validate responses.
        """
        secret_path = Path(tempfile.gettempdir()) / f".exo_node_secret_{self.node_id}"
        
        try:
            if secret_path.exists():
                with open(secret_path, "r") as f:
                    secret = f.read().strip()
                    if secret and len(secret) >= 32:
                        return secret
            
            # Generate a new secret
            new_secret = base64.b64encode(os.urandom(32)).decode("utf-8")
            
            # Save it securely
            with open(secret_path, "w") as f:
                f.write(new_secret)
            
            # Set file permissions to be readable only by the current user
            try:
                os.chmod(secret_path, 0o600)
            except OSError:
                # On Windows, this might fail, but we still have basic file security
                pass
                
            return new_secret
        except Exception as e:
            logging.error("Error creating node secret, falling back to in-memory secret",
                         component="auth",
                         exc_info=e)
            # Fallback to a runtime-only secret
            return base64.b64encode(os.urandom(32)).decode("utf-8")
    
    async def start(self):
        """Start the authentication system."""
        if not self.auth_enabled:
            logging.info("Peer authentication is disabled", component="auth")
            return
            
        # Load saved tokens
        self._load_tokens()
        
        # Start the token cleanup task
        self._shutting_down = False
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logging.info("Peer authentication system started", component="auth")
    
    async def stop(self):
        """Stop the authentication system."""
        if not self.auth_enabled:
            return
            
        self._shutting_down = True
        
        # Cancel the cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save tokens before shutdown
        self._save_tokens()
        
        logging.info("Peer authentication system stopped", component="auth")
    
    def _load_tokens(self):
        """Load authentication tokens from disk."""
        if not self.tokens_path.exists():
            return
            
        try:
            with open(self.tokens_path, "r") as f:
                data = json.load(f)
                
            current_time = time.time()
            loaded_count = 0
            expired_count = 0
            
            for token_data in data.get("tokens", []):
                token = AuthToken.from_dict(token_data)
                
                # Skip expired tokens
                if token.is_expired(current_time):
                    expired_count += 1
                    continue
                    
                self.tokens[token.token_id] = token
                loaded_count += 1
                
                # Add peer to authorized list if it's in the metadata
                peer_id = token.metadata.get("peer_id")
                if peer_id:
                    self.authorized_peers.add(peer_id)
            
            logging.info(f"Loaded authentication tokens", 
                        component="auth",
                        stats={"loaded": loaded_count, "expired": expired_count})
        except Exception as e:
            logging.error("Error loading authentication tokens",
                         component="auth",
                         exc_info=e)
    
    def _save_tokens(self):
        """Save authentication tokens to disk."""
        try:
            # Filter out expired tokens before saving
            current_time = time.time()
            valid_tokens = [token for token in self.tokens.values() if not token.is_expired(current_time)]
            
            data = {
                "tokens": [token.to_dict() for token in valid_tokens]
            }
            
            with open(self.tokens_path, "w") as f:
                json.dump(data, f, indent=2)
                
            # Set file permissions to be readable only by the current user
            try:
                os.chmod(self.tokens_path, 0o600)
            except OSError:
                # On Windows, this might fail, but we still have basic file security
                pass
                
            logging.debug(f"Saved {len(valid_tokens)} authentication tokens",
                         component="auth")
        except Exception as e:
            logging.error("Error saving authentication tokens",
                         component="auth",
                         exc_info=e)
    
    async def _cleanup_loop(self):
        """Periodically clean up expired tokens and challenges."""
        try:
            while not self._shutting_down:
                await asyncio.sleep(300)  # Run every 5 minutes
                self._cleanup_expired()
        except asyncio.CancelledError:
            logging.debug("Authentication cleanup task cancelled", component="auth")
            raise
        except Exception as e:
            logging.error("Error in authentication cleanup task", 
                         component="auth",
                         exc_info=e)
    
    def _cleanup_expired(self):
        """Clean up expired tokens and challenges."""
        current_time = time.time()
        
        # Clean up expired tokens
        expired_tokens = []
        for token_id, token in list(self.tokens.items()):
            if token.is_expired(current_time):
                expired_tokens.append(token_id)
                # Also remove from authorized peers if it's a peer token
                peer_id = token.metadata.get("peer_id")
                if peer_id:
                    self.authorized_peers.discard(peer_id)
                del self.tokens[token_id]
        
        # Clean up expired challenges
        expired_challenges = []
        for challenge_id, challenge in list(self.challenges.items()):
            if challenge.is_expired(current_time):
                expired_challenges.append(challenge_id)
                del self.challenges[challenge_id]
        
        if expired_tokens or expired_challenges:
            logging.debug("Cleaned up expired authentication data",
                         component="auth",
                         stats={
                             "expired_tokens": len(expired_tokens),
                             "expired_challenges": len(expired_challenges)
                         })
            
            # Save tokens after cleanup
            if expired_tokens:
                self._save_tokens()
    
    def generate_token(self, peer_id: str = None, expiry_seconds: int = None) -> AuthToken:
        """
        Generate a new authentication token.
        
        Args:
            peer_id: The ID of the peer this token is for (optional)
            expiry_seconds: Token validity period in seconds
            
        Returns:
            AuthToken object
        """
        if not self.auth_enabled:
            # Return a dummy token if auth is disabled
            return AuthToken("dummy", "dummy", time.time() + (24 * 60 * 60), {"peer_id": peer_id})
            
        # Use default expiry if not specified
        if expiry_seconds is None:
            expiry_seconds = self.token_expiry
            
        # Generate token data
        token_id = str(uuid.uuid4())
        token_secret = base64.b64encode(os.urandom(24)).decode("utf-8")
        expires_at = time.time() + expiry_seconds
        
        # Create metadata
        metadata = {}
        if peer_id:
            metadata["peer_id"] = peer_id
            metadata["issued_by"] = self.node_id
            metadata["issued_at"] = time.time()
        
        # Create the token
        token = AuthToken(token_id, token_secret, expires_at, metadata)
        
        # Store the token
        self.tokens[token_id] = token
        
        # Add peer to authorized list if peer_id was provided
        if peer_id:
            self.authorized_peers.add(peer_id)
            
        # Save updated tokens
        self._save_tokens()
        
        logging.info(f"Generated new authentication token",
                    component="auth",
                    token_id=token_id,
                    peer_id=peer_id,
                    expires_in=f"{expiry_seconds/3600:.1f} hours")
        
        return token
    
    def revoke_token(self, token_id: str) -> bool:
        """
        Revoke an authentication token.
        
        Args:
            token_id: The ID of the token to revoke
            
        Returns:
            True if the token was found and revoked, False otherwise
        """
        if not self.auth_enabled:
            return True
            
        if token_id in self.tokens:
            token = self.tokens[token_id]
            
            # Remove from authorized peers if it's a peer token
            peer_id = token.metadata.get("peer_id")
            if peer_id:
                self.authorized_peers.discard(peer_id)
            
            # Delete the token
            del self.tokens[token_id]
            
            # Save updated tokens
            self._save_tokens()
            
            logging.info(f"Revoked authentication token {token_id}", 
                        component="auth",
                        peer_id=peer_id)
            
            return True
        
        return False
    
    def create_challenge(self, peer_id: str) -> AuthChallenge:
        """
        Create an authentication challenge for a peer.
        
        Args:
            peer_id: The ID of the peer to challenge
            
        Returns:
            AuthChallenge object
        """
        if not self.auth_enabled:
            # Return a dummy challenge if auth is disabled
            challenge_id = str(uuid.uuid4())
            return AuthChallenge(
                challenge_id=challenge_id,
                challenge_data="dummy",
                node_id=self.node_id,
                created_at=time.time(),
                expires_at=time.time() + 60
            )
        
        # Generate challenge data
        challenge_id = str(uuid.uuid4())
        nonce = base64.b64encode(os.urandom(16)).decode("utf-8")
        timestamp = str(int(time.time()))
        challenge_data = f"{nonce}.{timestamp}"
        
        # Create the challenge
        challenge = AuthChallenge(
            challenge_id=challenge_id,
            challenge_data=challenge_data,
            node_id=self.node_id,
            created_at=time.time(),
            expires_at=time.time() + self.challenge_expiry
        )
        
        # Store the challenge
        self.challenges[challenge_id] = challenge
        
        logging.debug(f"Created authentication challenge for peer {peer_id}",
                     component="auth",
                     challenge_id=challenge_id)
        
        return challenge
    
    def create_challenge_response(self, challenge: AuthChallenge, token: AuthToken) -> AuthResponse:
        """
        Create a response to an authentication challenge.
        
        Args:
            challenge: The challenge to respond to
            token: The authentication token to use
            
        Returns:
            AuthResponse object
        """
        if not self.auth_enabled:
            # Return a dummy response if auth is disabled
            return AuthResponse(
                challenge_id=challenge.challenge_id,
                node_id=self.node_id,
                response_data="dummy",
                token_id=token.token_id
            )
        
        # Create response data (HMAC signature)
        signature = self._sign_data(
            data=f"{challenge.challenge_id}.{challenge.challenge_data}.{token.token_id}",
            key=token.token_secret
        )
        
        # Create the response
        response = AuthResponse(
            challenge_id=challenge.challenge_id,
            node_id=self.node_id,
            response_data=signature,
            token_id=token.token_id
        )
        
        logging.debug(f"Created authentication response for challenge {challenge.challenge_id}",
                     component="auth",
                     token_id=token.token_id)
        
        return response
    
    def verify_challenge_response(self, response: AuthResponse) -> bool:
        """
        Verify an authentication challenge response.
        
        Args:
            response: The challenge response to verify
            
        Returns:
            True if the response is valid, False otherwise
        """
        if not self.auth_enabled:
            return True
        
        # Check if the challenge exists
        if response.challenge_id not in self.challenges:
            logging.warning(f"Authentication failed: Challenge {response.challenge_id} not found",
                          component="auth",
                          node_id=response.node_id)
            return False
        
        challenge = self.challenges[response.challenge_id]
        
        # Check if the challenge has expired
        if challenge.is_expired():
            logging.warning(f"Authentication failed: Challenge {response.challenge_id} has expired",
                          component="auth",
                          node_id=response.node_id)
            del self.challenges[response.challenge_id]
            return False
        
        # Check if the token exists
        if response.token_id not in self.tokens:
            logging.warning(f"Authentication failed: Token {response.token_id} not found",
                          component="auth",
                          node_id=response.node_id)
            return False
        
        token = self.tokens[response.token_id]
        
        # Check if the token has expired
        if token.is_expired():
            logging.warning(f"Authentication failed: Token {response.token_id} has expired",
                          component="auth",
                          node_id=response.node_id)
            del self.tokens[response.token_id]
            self._save_tokens()
            return False
        
        # Check if the token is for the correct peer
        peer_id = token.metadata.get("peer_id")
        if peer_id and peer_id != response.node_id:
            logging.warning(f"Authentication failed: Token {response.token_id} belongs to peer {peer_id}, not {response.node_id}",
                          component="auth",
                          node_id=response.node_id)
            return False
        
        # Verify the signature
        expected_signature = self._sign_data(
            data=f"{challenge.challenge_id}.{challenge.challenge_data}.{token.token_id}",
            key=token.token_secret
        )
        
        if response.response_data != expected_signature:
            logging.warning(f"Authentication failed: Invalid signature for challenge {response.challenge_id}",
                          component="auth",
                          node_id=response.node_id)
            return False
        
        # Authentication successful
        
        # Add peer to authorized list
        self.authorized_peers.add(response.node_id)
        
        # Remove the used challenge
        del self.challenges[response.challenge_id]
        
        logging.info(f"Peer {response.node_id} authenticated successfully",
                    component="auth",
                    token_id=response.token_id)
        
        return True
    
    def is_peer_authorized(self, peer_id: str) -> bool:
        """
        Check if a peer is authorized.
        
        Args:
            peer_id: The ID of the peer to check
            
        Returns:
            True if the peer is authorized, False otherwise
        """
        if not self.auth_enabled:
            return True
            
        return peer_id in self.authorized_peers
    
    def _sign_data(self, data: str, key: str) -> str:
        """
        Create an HMAC signature for authentication.
        
        Args:
            data: The data to sign
            key: The key to use for signing
            
        Returns:
            Base64-encoded signature
        """
        # Convert inputs to bytes
        data_bytes = data.encode("utf-8")
        key_bytes = key.encode("utf-8")
        
        # Create HMAC-SHA256 signature
        signature = hmac.new(key_bytes, data_bytes, hashlib.sha256).digest()
        
        # Encode as base64
        return base64.b64encode(signature).decode("utf-8")


# Global authentication instance
_auth_instance: Optional[PeerAuthentication] = None


def get_auth_instance(node_id: str = None) -> PeerAuthentication:
    """
    Get the global authentication instance.
    
    Args:
        node_id: The ID of this node (required when creating the instance)
        
    Returns:
        PeerAuthentication instance
    """
    global _auth_instance
    
    if _auth_instance is None:
        if node_id is None:
            raise ValueError("node_id is required when creating the authentication instance")
            
        # Check if authentication is enabled
        auth_enabled = os.environ.get("EXO_AUTH_ENABLED", "1").lower() in ("1", "true", "yes")
        
        _auth_instance = PeerAuthentication(node_id, auth_enabled=auth_enabled)
        
    return _auth_instance


async def initialize_auth(node_id: str) -> PeerAuthentication:
    """
    Initialize the authentication system.
    
    Args:
        node_id: The ID of this node
        
    Returns:
        PeerAuthentication instance
    """
    auth = get_auth_instance(node_id)
    await auth.start()
    return auth


async def shutdown_auth() -> None:
    """Shut down the authentication system."""
    global _auth_instance
    
    if _auth_instance:
        await _auth_instance.stop()
        _auth_instance = None