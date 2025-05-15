from exo.inference.shard import Shard
from exo.models import get_repo
from pathlib import Path
from exo.download.hf.hf_helpers import get_hf_endpoint, get_auth_headers, filter_repo_objects, get_allow_patterns
from exo.download.shard_download import ShardDownloader
from exo.download.download_progress import RepoProgressEvent, RepoFileProgressEvent
from exo.helpers import AsyncCallbackSystem, DEBUG
from exo.models import get_supported_models, build_full_shard
import os
import aiofiles.os as aios
import aiohttp
import aiofiles
from urllib.parse import urljoin
from typing import Callable, Union, Tuple, Dict, List, Optional, Literal, AsyncIterator
import time
from datetime import timedelta
import asyncio
import json
import traceback
import shutil

# Dictionary to store locks for model deletion
_model_deletion_locks = {}

# Dictionary to store locks for file download to prevent race conditions
_file_download_locks = {}
import tempfile
import hashlib

def exo_home() -> Path:
  return Path(os.environ.get("EXO_HOME", Path.home()/".cache"/"exo"))

def exo_tmp() -> Path:
  return Path(tempfile.gettempdir())/"exo"

async def ensure_exo_home() -> Path:
  await aios.makedirs(exo_home(), exist_ok=True)
  return exo_home()

async def ensure_exo_tmp() -> Path:
  await aios.makedirs(exo_tmp(), exist_ok=True)
  return exo_tmp()

async def has_exo_home_read_access() -> bool:
  try: return await aios.access(exo_home(), os.R_OK)
  except OSError: return False

async def has_exo_home_write_access() -> bool:
  try: return await aios.access(exo_home(), os.W_OK)
  except OSError: return False

async def ensure_downloads_dir() -> Path:
  downloads_dir = exo_home()/"downloads"
  await aios.makedirs(downloads_dir, exist_ok=True)
  return downloads_dir

async def delete_model(model_id: str, inference_engine_name: str) -> bool:
  repo_id = get_repo(model_id, inference_engine_name)
  lock_key = f"{model_id}_{inference_engine_name}"

  # Create lock for this model if it doesn't exist
  if lock_key not in _model_deletion_locks:
    _model_deletion_locks[lock_key] = asyncio.Lock()

  # Acquire lock before performing deletion
  async with _model_deletion_locks[lock_key]:
    model_dir = await ensure_downloads_dir()/repo_id.replace("/", "--")
    if not await aios.path.exists(model_dir):
      return False

    try:
      await asyncio.to_thread(shutil.rmtree, model_dir, ignore_errors=False)
      if DEBUG >= 2:
        print(f"Successfully deleted model directory: {model_dir}")
      return True
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error deleting model directory {model_dir}: {e}")
      return False

async def seed_models(seed_dir: Union[str, Path]):
  """
  Move model from a source directory to the exo downloads cache directory.

  Args:
      seed_dir: Source directory containing model directories (must be a valid path)

  Returns:
      None

  Raises:
      FileNotFoundError: If the source directory doesn't exist
  """
  if not seed_dir:
    raise ValueError("seed_dir must be provided")

  source_dir = Path(seed_dir)

  # Check if source directory exists
  if not source_dir.exists():
    raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

  if not source_dir.is_dir():
    raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

  dest_dir = await ensure_downloads_dir()

  if DEBUG >= 2:
    print(f"Seeding models from {source_dir} to {dest_dir}")

  # Track successful and failed operations
  successful = []
  failed = []

  try:
    # First check if we can iterate through the directory
    paths = list(source_dir.iterdir())

    for path in paths:
      if path.is_dir() and path.name.startswith("models--"):
        dest_path = dest_dir/path.name

        if await aios.path.exists(dest_path):
          if DEBUG >= 1:
            print(f'Skipping {path.name}: Destination already exists')
          continue

        try:
          # First try renaming (faster but only works on same filesystem)
          try:
            await aios.rename(str(path), str(dest_path))
            successful.append(path.name)
            if DEBUG >= 1:
              print(f"Successfully moved model {path.name}")
          except OSError:
            # If rename fails (different filesystems), try copy+delete
            if DEBUG >= 1:
              print(f"Rename failed for {path}, trying copy+delete")
            await asyncio.to_thread(shutil.copytree, path, dest_path)
            await asyncio.to_thread(shutil.rmtree, path)
            successful.append(path.name)
            if DEBUG >= 1:
              print(f"Successfully copied model {path.name}")
        except PermissionError as e:
          failed.append((path.name, f"Permission error: {str(e)}"))
          if DEBUG >= 0:
            print(f"Permission error seeding model {path} to {dest_path}: {e}")
        except shutil.Error as e:
          failed.append((path.name, f"Copy error: {str(e)}"))
          if DEBUG >= 0:
            print(f"Copy error seeding model {path} to {dest_path}: {e}")
        except Exception as e:
          failed.append((path.name, f"Error: {str(e)}"))
          if DEBUG >= 0:
            print(f"Error seeding model {path} to {dest_path}: {e}")
            traceback.print_exc()

  except PermissionError as e:
    if DEBUG >= 0:
      print(f"Permission error accessing source directory {source_dir}: {e}")
    raise
  except Exception as e:
    if DEBUG >= 0:
      print(f"Error during model seeding from {source_dir}: {e}")
      traceback.print_exc()
    raise

  # Summary
  if DEBUG >= 1:
    if successful:
      print(f"Successfully seeded {len(successful)} models: {', '.join(successful)}")
    if failed:
      print(f"Failed to seed {len(failed)} models:")
      for name, error in failed:
        print(f"  - {name}: {error}")

async def fetch_file_list_with_cache(repo_id: str, revision: str = "main") -> List[Dict[str, Union[str, int]]]:
  cache_file = (await ensure_exo_tmp())/f"{repo_id.replace('/', '--')}--{revision}--file_list.json"
  if await aios.path.exists(cache_file):
    async with aiofiles.open(cache_file, 'r') as f: return json.loads(await f.read())
  file_list = await fetch_file_list_with_retry(repo_id, revision)
  await aios.makedirs(cache_file.parent, exist_ok=True)
  async with aiofiles.open(cache_file, 'w') as f: await f.write(json.dumps(file_list))
  return file_list

async def fetch_file_list_with_retry(repo_id: str, revision: str = "main", path: str = "") -> List[Dict[str, Union[str, int]]]:
  """
  Fetch a file list with retries and exponential backoff.
  
  Args:
      repo_id: Repository ID to fetch files from
      revision: Repository revision (branch/tag/commit)
      path: Path within the repository to start from
      
  Returns:
      List of file information dictionaries with path and size
      
  Raises:
      Exception: If the file list cannot be fetched after all retries
  """
  n_attempts = 30
  retryable_status_codes = [429, 500, 502, 503, 504]
  
  for attempt in range(n_attempts):
    try: 
      return await _fetch_file_list(repo_id, revision, path)
    except aiohttp.ClientResponseError as e:
      # Check if the status code is retryable
      if e.status not in retryable_status_codes or attempt == n_attempts - 1: 
        raise e
        
      wait_time = min(8, 0.1 * (2 ** attempt))
      if DEBUG >= 1:
        print(f"API returned status {e.status} when fetching file list for {repo_id}, " 
              f"retrying in {wait_time:.2f}s (attempt {attempt+1}/{n_attempts})")
      await asyncio.sleep(wait_time)
    except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
      # Network-related errors
      if attempt == n_attempts - 1: 
        raise Exception(f"Failed to connect to API for {repo_id} after {n_attempts} attempts: {e}") from e
        
      wait_time = min(8, 0.1 * (2 ** attempt))
      if DEBUG >= 1:
        print(f"Connection error when fetching file list for {repo_id}, " 
              f"retrying in {wait_time:.2f}s (attempt {attempt+1}/{n_attempts}): {e}")
      await asyncio.sleep(wait_time)
    except Exception as e:
      # Other unexpected errors
      if attempt == n_attempts - 1: 
        raise e
        
      wait_time = min(8, 0.1 * (2 ** attempt))
      if DEBUG >= 1:
        print(f"Error fetching file list for {repo_id}, "
              f"retrying in {wait_time:.2f}s (attempt {attempt+1}/{n_attempts}): {e}")
      await asyncio.sleep(wait_time)

async def _fetch_file_list(repo_id: str, revision: str = "main", path: str = "") -> List[Dict[str, Union[str, int]]]:
  """
  Internal helper to fetch the file list from the repository API.
  
  Args:
      repo_id: Repository ID to fetch files from
      revision: Repository revision (branch/tag/commit)
      path: Path within the repository to start from
      
  Returns:
      List of file information dictionaries with path and size
      
  Raises:
      Exception: If the API returns an error or an unexpected response
  """
  api_url = f"{get_hf_endpoint()}/api/models/{repo_id}/tree/{revision}"
  url = f"{api_url}/{path}" if path else api_url

  headers = await get_auth_headers()
  
  # Use increasing timeouts based on path depth to handle large directories
  # Base directories need more time as they can contain many files/subdirectories
  timeout_factor = 1 + path.count('/') * 0.5  # Increase timeout for deeper paths
  timeout = aiohttp.ClientTimeout(
      total=30 * timeout_factor, 
      connect=10, 
      sock_read=30 * timeout_factor, 
      sock_connect=10
  )
  
  try:
    async with aiohttp.ClientSession(timeout=timeout) as session:
      async with session.get(url, headers=headers, raise_for_status=True) as response:
        data = await response.json()
        files = []
        
        for item in data:
          if item["type"] == "file":
            files.append({"path": item["path"], "size": item["size"]})
          elif item["type"] == "directory":
            try:
              # Fetch subdirectory contents
              subfiles = await _fetch_file_list(repo_id, revision, item["path"])
              files.extend(subfiles)
            except Exception as e:
              # Log directory errors but continue with other items
              if DEBUG >= 1:
                print(f"Error fetching directory {item['path']} in {repo_id}: {e}")
                if DEBUG >= 2:
                  traceback.print_exc()
                  
        return files
  except aiohttp.ClientResponseError as e:
    # Provide more context in the error message
    status_msg = f"{e.status} ({e.message})" if hasattr(e, 'message') else e.status
    raise aiohttp.ClientResponseError(
        request_info=e.request_info,
        history=e.history,
        status=e.status,
        message=f"API error fetching file list for {repo_id}/{path}: {status_msg}",
        headers=e.headers
    )
  except Exception as e:
    # Wrap other exceptions with context
    raise Exception(f"Error fetching file list for {repo_id}/{path}: {e}") from e

async def calc_hash(path: Path, type: Literal["sha1", "sha256"] = "sha1") -> str:
  hash = hashlib.sha1() if type == "sha1" else hashlib.sha256()
  if type == "sha1":
    header = f"blob {(await aios.stat(path)).st_size}\0".encode()
    hash.update(header)
  async with aiofiles.open(path, 'rb') as f:
    while chunk := await f.read(8 * 1024 * 1024):
      hash.update(chunk)
  return hash.hexdigest()

async def file_meta(repo_id: str, revision: str, path: str) -> Tuple[int, str]:
  url = urljoin(f"{get_hf_endpoint()}/{repo_id}/resolve/{revision}/", path)
  headers = await get_auth_headers()
  async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800, connect=60, sock_read=1800, sock_connect=60)) as session:
    async with session.head(url, headers=headers) as r:
      content_length = int(r.headers.get('x-linked-size') or r.headers.get('content-length') or 0)
      etag = r.headers.get('X-Linked-ETag') or r.headers.get('ETag') or r.headers.get('Etag')
      assert content_length > 0, f"No content length for {url}"
      assert etag is not None, f"No remote hash for {url}"
      if  (etag[0] == '"' and etag[-1] == '"') or (etag[0] == "'" and etag[-1] == "'"): etag = etag[1:-1]
      return content_length, etag

async def download_file_with_retry(repo_id: str, revision: str, path: str, target_dir: Path, on_progress: Callable[[int, int], None] = lambda _, __: None) -> Path:
  n_attempts = 30
  for attempt in range(n_attempts):
    try: return await _download_file(repo_id, revision, path, target_dir, on_progress)
    except Exception as e:
      if isinstance(e, FileNotFoundError) or attempt == n_attempts - 1: raise e
      print(f"Download error on attempt {attempt}/{n_attempts} for {repo_id=} {revision=} {path=} {target_dir=}")
      traceback.print_exc()
      await asyncio.sleep(min(8, 0.1 * (2 ** attempt)))

async def _download_file(repo_id: str, revision: str, path: str, target_dir: Path, on_progress: Callable[[int, int], None] = lambda _, __: None) -> Path:
  # Generate a unique key for this file download
  lock_key = f"{repo_id}_{revision}_{path}"

  # Create a lock for this download if it doesn't exist
  if lock_key not in _file_download_locks:
    _file_download_locks[lock_key] = asyncio.Lock()

  # Acquire the lock to prevent concurrent downloads of the same file
  async with _file_download_locks[lock_key]:
    # Check if file exists again after acquiring the lock
    # Another process may have completed the download while waiting for the lock
    if await aios.path.exists(target_dir/path):
      if DEBUG >= 2:
        print(f"File already exists after acquiring lock: {target_dir/path}")
      return target_dir/path

    # Create parent directories for target file
    await aios.makedirs((target_dir/path).parent, exist_ok=True)
    
    # Get file metadata (size and hash)
    try:
      length, etag = await file_meta(repo_id, revision, path)
      remote_hash = etag[:-5] if etag.endswith("-gzip") else etag
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error getting file metadata for {repo_id}/{revision}/{path}: {e}")
      # Retry with exponential backoff
      await asyncio.sleep(2)
      length, etag = await file_meta(repo_id, revision, path)
      remote_hash = etag[:-5] if etag.endswith("-gzip") else etag
      
    # Path for regular partial file and our unique temporary partial file
    partial_path = target_dir/f"{path}.partial"
    unique_partial_path = target_dir/f"{path}.partial.{os.getpid()}.{int(time.time() * 1000)}"

    # If regular partial file exists, check if we can resume
    resume_byte_pos = None
    if await aios.path.exists(partial_path):
      try:
        resume_byte_pos = (await aios.stat(partial_path)).st_size
        if resume_byte_pos >= length:
          if DEBUG >= 2:
            print(f"Existing partial file is complete or larger than expected: {partial_path}")
          # If the file is complete, just rename it to the final destination
          if resume_byte_pos == length:
            try:
              await aios.rename(partial_path, target_dir/path)
              return target_dir/path
            except Exception as e:
              if DEBUG >= 1:
                print(f"Error renaming complete partial file {partial_path}: {e}")
          # If the file is larger than expected, discard it
          resume_byte_pos = None
        else:
          if DEBUG >= 2:
            print(f"Resuming download from byte {resume_byte_pos}: {partial_path}")
          # Copy existing partial file to our unique path to resume
          await asyncio.to_thread(shutil.copy2, partial_path, unique_partial_path)
      except Exception as e:
        if DEBUG >= 1:
          print(f"Error checking existing partial file {partial_path}: {e}")
        resume_byte_pos = None

    # Download the file if needed
    if resume_byte_pos != length:
      url = urljoin(f"{get_hf_endpoint()}/{repo_id}/resolve/{revision}/", path)
      
      # Use tiered timeout and retry strategy for download
      max_attempts = 3
      connection_timeouts = [30, 60, 90]  # Increase timeout for each retry
      read_timeouts = [300, 600, 1800]    # Increase timeout for each retry
      
      for attempt in range(max_attempts):
        try:
          headers = await get_auth_headers()
          if resume_byte_pos: 
            headers['Range'] = f'bytes={resume_byte_pos}-'
          n_read = resume_byte_pos or 0

          # Create connection with timeouts appropriate for the current attempt
          connect_timeout = connection_timeouts[min(attempt, len(connection_timeouts)-1)]
          read_timeout = read_timeouts[min(attempt, len(read_timeouts)-1)]
          
          if DEBUG >= 2 and attempt > 0:
            print(f"Attempt {attempt+1}/{max_attempts} downloading {path}: connect_timeout={connect_timeout}s, read_timeout={read_timeout}s")
            
          # Configure timeout for this attempt
          timeout = aiohttp.ClientTimeout(
            total=read_timeout + connect_timeout, 
            connect=connect_timeout,
            sock_read=read_timeout,
            sock_connect=connect_timeout
          )

          async with aiohttp.ClientSession(timeout=timeout) as session:
            # Set TCP keepalive to detect connection drops
            if hasattr(session, "_connector"):
              if hasattr(session._connector, "_tcp_keepalive"):
                session._connector._tcp_keepalive = True
                
            # Use the same timeout for the GET request
            async with session.get(url, headers=headers, timeout=timeout) as r:
              if r.status == 404: 
                raise FileNotFoundError(f"File not found: {url}")
                
              if r.status not in [200, 206]:
                if attempt < max_attempts - 1:
                  # If it's a retryable status code, retry after delay
                  retryable_codes = [429, 500, 502, 503, 504]
                  if r.status in retryable_codes:
                    wait_time = 2 ** attempt  # Exponential backoff
                    if DEBUG >= 1:
                      print(f"Received status {r.status} for {url}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                    
                # Either non-retryable code or final attempt
                raise Exception(f"Failed to download {path} from {url}: HTTP {r.status}")

              # Write to our unique partial file
              async with aiofiles.open(unique_partial_path, 'ab' if resume_byte_pos else 'wb') as f:
                # Larger chunks for faster downloads, but not too large
                chunk_size = 8 * 1024 * 1024  # 8MB chunks
                
                # Use while loop with explicit read to handle connection issues better
                while True:
                  try:
                    chunk = await r.content.read(chunk_size)
                    if not chunk:  # End of file
                      break
                      
                    bytes_written = await f.write(chunk)
                    n_read += bytes_written
                    on_progress(n_read, length)
                    
                  except asyncio.TimeoutError:
                    if DEBUG >= 1:
                      print(f"Timeout reading chunk for {url} at position {n_read}/{length}")
                    if attempt < max_attempts - 1:
                      # Set resume position for next attempt
                      resume_byte_pos = n_read
                      await asyncio.sleep(2 ** attempt)  # Exponential backoff
                      break
                    raise
                      
                  except Exception as e:
                    if DEBUG >= 1:
                      print(f"Error reading chunk for {url}: {e}")
                    if attempt < max_attempts - 1:
                      # Set resume position for next attempt
                      resume_byte_pos = n_read
                      await asyncio.sleep(2 ** attempt)  # Exponential backoff
                      break
                    raise
                    
              # If we've read the full file, break out of retry loop
              if n_read >= length:
                break
                
        except FileNotFoundError:
          # Don't retry file not found errors
          raise
          
        except Exception as e:
          if DEBUG >= 1:
            print(f"Error downloading {url} (attempt {attempt+1}/{max_attempts}): {e}")
            if DEBUG >= 2:
              traceback.print_exc()
              
          if attempt < max_attempts - 1:
            # Wait with exponential backoff before retry
            wait_time = 2 ** attempt
            await asyncio.sleep(wait_time)
            continue
          # On final attempt, propagate the exception
          raise Exception(f"Failed to download {path} after {max_attempts} attempts: {e}")

    # Use the unique partial path for verification
    final_path = unique_partial_path
    if not await aios.path.exists(final_path):
      raise FileNotFoundError(f"Partial file missing after download: {final_path}")

    # Verify file integrity
    try:
      final_hash = await calc_hash(final_path, type="sha256" if len(remote_hash) == 64 else "sha1")
      integrity = final_hash == remote_hash

      if not integrity:
        if DEBUG >= 1:
          print(f"Hash mismatch for {path}: got {final_hash}, expected {remote_hash}")
          
        # Try up to 2 times to re-download with verification
        for retry in range(2):
          try:
            # Remove corrupt file
            await aios.remove(final_path)
            if DEBUG >= 1:
              print(f"Removed corrupted file, retrying download for {path} (attempt {retry+1}/2)")
              
            # Start fresh (no resume)
            resume_byte_pos = None
            unique_partial_path = target_dir/f"{path}.partial.{os.getpid()}.{int(time.time() * 1000)}"
            
            # Attempt to download again with full verification
            url = urljoin(f"{get_hf_endpoint()}/{repo_id}/resolve/{revision}/", path)
            headers = await get_auth_headers()
            n_read = 0
            
            # Use longer timeouts for retries
            timeout = aiohttp.ClientTimeout(total=1800, connect=90, sock_read=1800, sock_connect=90)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
              async with session.get(url, headers=headers, timeout=timeout) as r:
                if r.status != 200:
                  raise Exception(f"Failed to download {path} on integrity retry: HTTP {r.status}")
                  
                async with aiofiles.open(unique_partial_path, 'wb') as f:
                  while chunk := await r.content.read(8 * 1024 * 1024):
                    on_progress(n_read := n_read + await f.write(chunk), length)
            
            # Verify the hash again
            final_hash = await calc_hash(unique_partial_path, type="sha256" if len(remote_hash) == 64 else "sha1")
            integrity = final_hash == remote_hash
            
            if integrity:
              final_path = unique_partial_path
              break
            else:
              if DEBUG >= 1:
                print(f"Hash still mismatched on retry {retry+1}: {final_hash} vs {remote_hash}")
          except Exception as e:
            if DEBUG >= 1:
              print(f"Error during integrity retry {retry+1}: {e}")
            if retry == 1:  # Last retry
              raise Exception(f"Failed to download {path} with correct hash after multiple attempts")
            
        # If we still don't have integrity after retries, fail
        if not integrity:
          raise Exception(f"Downloaded file {target_dir/path} has hash {final_hash} but remote hash is {remote_hash}")
    except Exception as e:
      # Clean up on verification failure
      try:
        if await aios.path.exists(final_path):
          await aios.remove(final_path)
      except:
        pass
      raise e

    # Atomic move to the final destination
    try:
      await aios.rename(final_path, target_dir/path)
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error renaming {final_path} to {target_dir/path}: {e}")
      # Try an alternative approach if rename fails
      try:
        await asyncio.to_thread(shutil.copy2, final_path, target_dir/path)
        await aios.remove(final_path)
      except Exception as copy_e:
        if DEBUG >= 1:
          print(f"Alternative copy approach also failed: {copy_e}")
        raise e  # Raise the original error

    # Clean up any leftover partial files
    try:
      if await aios.path.exists(partial_path):
        await aios.remove(partial_path)
        if DEBUG >= 2:
          print(f"Removed old partial file {partial_path}")
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error cleaning up partial file {partial_path}: {e}")

    if DEBUG >= 2:
      print(f"Successfully downloaded and verified {target_dir/path}")

    return target_dir/path


def calculate_repo_progress(shard: Shard, repo_id: str, revision: str, file_progress: Dict[str, RepoFileProgressEvent], all_start_time: float) -> RepoProgressEvent:
  all_total_bytes = sum([p.total for p in file_progress.values()])
  all_downloaded_bytes = sum([p.downloaded for p in file_progress.values()])
  all_downloaded_bytes_this_session = sum([p.downloaded_this_session for p in file_progress.values()])
  elapsed_time = time.time() - all_start_time
  all_speed = all_downloaded_bytes_this_session / elapsed_time if elapsed_time > 0 else 0
  all_eta = timedelta(seconds=(all_total_bytes - all_downloaded_bytes) / all_speed) if all_speed > 0 else timedelta(seconds=0)
  status = "complete" if all(p.status == "complete" for p in file_progress.values()) else "in_progress" if any(p.status == "in_progress" for p in file_progress.values()) else "not_started"
  return RepoProgressEvent(shard, repo_id, revision, len([p for p in file_progress.values() if p.downloaded == p.total]), len(file_progress), all_downloaded_bytes, all_downloaded_bytes_this_session, all_total_bytes, all_speed, all_eta, file_progress, status)

async def get_weight_map(repo_id: str, revision: str = "main") -> Dict[str, str]:
  target_dir = (await ensure_exo_tmp())/repo_id.replace("/", "--")
  index_file = await download_file_with_retry(repo_id, revision, "model.safetensors.index.json", target_dir)
  async with aiofiles.open(index_file, 'r') as f: index_data = json.loads(await f.read())
  return index_data.get("weight_map")

async def resolve_allow_patterns(shard: Shard, inference_engine_classname: str) -> List[str]:
  """
  Resolves the allowed file patterns for a specific model shard and inference engine.

  Args:
      shard: The model shard to resolve patterns for
      inference_engine_classname: Name of the inference engine class

  Returns:
      List of file patterns to include in the download
  """
  default_patterns = ["*"]  # Default fallback patterns if we can't get specific ones

  if not shard or not shard.model_id:
    if DEBUG >= 1:
      print(f"Invalid shard provided: {shard}")
    return default_patterns

  try:
    # Get the repository ID for this model and engine
    repo_id = get_repo(shard.model_id, inference_engine_classname)
    if not repo_id:
      if DEBUG >= 1:
        print(f"No repository found for {shard.model_id=} and inference engine {inference_engine_classname}")
      return default_patterns

    # Get the weight map from the model index file
    weight_map = await get_weight_map(repo_id)
    if not weight_map:
      if DEBUG >= 1:
        print(f"Empty or missing weight map for {shard.model_id=}")
      return default_patterns

    # Get the specific patterns for this shard from the weight map
    patterns = get_allow_patterns(weight_map, shard)
    if not patterns:
      if DEBUG >= 1:
        print(f"No patterns found for {shard=} in weight map")
      return default_patterns

    return patterns

  except FileNotFoundError as e:
    if DEBUG >= 1:
      print(f"Model index file not found for {shard.model_id=}: {e}")
    return default_patterns

  except json.JSONDecodeError as e:
    if DEBUG >= 1:
      print(f"Invalid JSON in model index file for {shard.model_id=}: {e}")
    return default_patterns

  except ValueError as e:
    if DEBUG >= 1:
      print(f"Value error resolving patterns for {shard.model_id=}: {e}")
    return default_patterns

  except Exception as e:
    if DEBUG >= 1:
      print(f"Unexpected error getting weight map for {shard.model_id=} and inference engine {inference_engine_classname}: {e}")
      traceback.print_exc()
    return default_patterns

async def get_downloaded_size(path: Path) -> int:
  partial_path = path.with_suffix(path.suffix + ".partial")
  if await aios.path.exists(path): return (await aios.stat(path)).st_size
  if await aios.path.exists(partial_path): return (await aios.stat(partial_path)).st_size
  return 0

async def download_shard(shard: Shard, inference_engine_classname: str, on_progress: AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]], max_parallel_downloads: int = 8, skip_download: bool = False) -> tuple[Path, RepoProgressEvent]:
  if DEBUG >= 2 and not skip_download: print(f"Downloading {shard.model_id=} for {inference_engine_classname}")
  repo_id = get_repo(shard.model_id, inference_engine_classname)
  revision = "main"
  target_dir = await ensure_downloads_dir()/repo_id.replace("/", "--")
  if not skip_download: await aios.makedirs(target_dir, exist_ok=True)

  if repo_id is None:
    raise ValueError(f"No repo found for {shard.model_id=} and inference engine {inference_engine_classname}")

  allow_patterns = await resolve_allow_patterns(shard, inference_engine_classname)
  if DEBUG >= 2: print(f"Downloading {shard.model_id=} with {allow_patterns=}")

  all_start_time = time.time()
  file_list = await fetch_file_list_with_cache(repo_id, revision)
  filtered_file_list = list(filter_repo_objects(file_list, allow_patterns=allow_patterns, key=lambda x: x["path"]))
  file_progress: Dict[str, RepoFileProgressEvent] = {}
  async def on_progress_wrapper(file: dict, curr_bytes: int, total_bytes: int):
    start_time = file_progress[file["path"]].start_time if file["path"] in file_progress else time.time()
    downloaded_this_session = file_progress[file["path"]].downloaded_this_session + (curr_bytes - file_progress[file["path"]].downloaded) if file["path"] in file_progress else curr_bytes
    speed = downloaded_this_session / (time.time() - start_time) if time.time() - start_time > 0 else 0
    eta = timedelta(seconds=(total_bytes - curr_bytes) / speed) if speed > 0 else timedelta(seconds=0)
    file_progress[file["path"]] = RepoFileProgressEvent(repo_id, revision, file["path"], curr_bytes, downloaded_this_session, total_bytes, speed, eta, "complete" if curr_bytes == total_bytes else "in_progress", start_time)
    repo_progress = calculate_repo_progress(shard, repo_id, revision, file_progress, all_start_time)
    on_progress.trigger_all(shard, repo_progress)
    if DEBUG >= 6: print(f"Downloading {file['path']} {curr_bytes}/{total_bytes} {speed} {eta}")
  
  # Create a synchronous wrapper that can be called from non-async contexts
  def sync_progress_wrapper(file: dict, curr_bytes: int, total_bytes: int):
    asyncio.create_task(on_progress_wrapper(file, curr_bytes, total_bytes))
  for file in filtered_file_list:
    downloaded_bytes = await get_downloaded_size(target_dir/file["path"])
    file_progress[file["path"]] = RepoFileProgressEvent(repo_id, revision, file["path"], downloaded_bytes, 0, file["size"], 0, timedelta(0), "complete" if downloaded_bytes == file["size"] else "not_started", time.time())

  semaphore = asyncio.Semaphore(max_parallel_downloads)
  async def download_with_semaphore(file):
    async with semaphore:
      await download_file_with_retry(repo_id, revision, file["path"], target_dir, lambda curr_bytes, total_bytes: sync_progress_wrapper(file, curr_bytes, total_bytes))
  if not skip_download: await asyncio.gather(*[download_with_semaphore(file) for file in filtered_file_list])
  final_repo_progress = calculate_repo_progress(shard, repo_id, revision, file_progress, all_start_time)
  on_progress.trigger_all(shard, final_repo_progress)
  if gguf := next((f for f in filtered_file_list if f["path"].endswith(".gguf")), None):
    return target_dir/gguf["path"], final_repo_progress
  else:
    return target_dir, final_repo_progress

def new_shard_downloader(max_parallel_downloads: int = 8) -> ShardDownloader:
  return SingletonShardDownloader(CachedShardDownloader(NewShardDownloader(max_parallel_downloads)))

class SingletonShardDownloader(ShardDownloader):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard_downloader = shard_downloader
    self.active_downloads: Dict[Shard, asyncio.Task] = {}
    self._lock = asyncio.Lock()  # Add a lock for thread safety

  @property
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    return self.shard_downloader.on_progress

  async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
    # Use a lock to prevent race conditions when checking/creating tasks
    async with self._lock:
      # Check if we already have an active download for this shard
      if shard not in self.active_downloads or self.active_downloads[shard].done():
        # Create a new task if none exists or the previous one is done
        if DEBUG >= 2:
          print(f"Creating new download task for {shard.model_id} ({inference_engine_name})")
        self.active_downloads[shard] = asyncio.create_task(
          self.shard_downloader.ensure_shard(shard, inference_engine_name)
        )
      else:
        if DEBUG >= 2:
          print(f"Reusing existing download task for {shard.model_id} ({inference_engine_name})")

      # Get the task reference outside the lock to avoid holding the lock during await
      task = self.active_downloads[shard]

    try:
      # Wait for the task to complete
      result = await task
      return result
    finally:
      # Clean up completed tasks under the lock to prevent race conditions
      async with self._lock:
        if shard in self.active_downloads and self.active_downloads[shard].done():
          if DEBUG >= 2:
            print(f"Cleaning up completed download task for {shard.model_id}")
          del self.active_downloads[shard]

  async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
    async for path, status in self.shard_downloader.get_shard_download_status(inference_engine_name):
      yield path, status

class CachedShardDownloader(ShardDownloader):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard_downloader = shard_downloader
    self.cache: Dict[tuple[str, Shard], Path] = {}

  @property
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    return self.shard_downloader.on_progress

  async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
    if (inference_engine_name, shard) in self.cache:
      if DEBUG >= 2: print(f"ensure_shard cache hit {shard=} for {inference_engine_name}")
      return self.cache[(inference_engine_name, shard)]
    if DEBUG >= 2: print(f"ensure_shard cache miss {shard=} for {inference_engine_name}")
    target_dir = await self.shard_downloader.ensure_shard(shard, inference_engine_name)
    self.cache[(inference_engine_name, shard)] = target_dir
    return target_dir

  async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
    async for path, status in self.shard_downloader.get_shard_download_status(inference_engine_name):
      yield path, status

class NewShardDownloader(ShardDownloader):
  def __init__(self, max_parallel_downloads: int = 8, download_chunk_size: int = 8 * 1024 * 1024):
    """
    Initialize the NewShardDownloader with configurable parameters.

    Args:
        max_parallel_downloads: Maximum number of parallel downloads allowed
        download_chunk_size: Size of chunks to read when downloading files (in bytes)
    """
    self.max_parallel_downloads = max_parallel_downloads
    self.download_chunk_size = download_chunk_size
    self._on_progress = AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]()

  @property
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    return self._on_progress

  async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
    target_dir, _ = await download_shard(
      shard,
      inference_engine_name,
      self.on_progress,
      max_parallel_downloads=self.max_parallel_downloads
    )
    return target_dir

  async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
    if DEBUG >= 2: print("Getting shard download status for", inference_engine_name)
    tasks = [download_shard(
      build_full_shard(model_id, inference_engine_name),
      inference_engine_name,
      self.on_progress,
      skip_download=True
    ) for model_id in get_supported_models([[inference_engine_name]])]
    for task in asyncio.as_completed(tasks):
      try:
        path, progress = await task
        yield (path, progress)
      except Exception as e:
        print("Error downloading shard:", e)
