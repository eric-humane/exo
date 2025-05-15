import os
import sys
import asyncio
from typing import Callable, TypeVar, Optional, Dict, Generic, Tuple, List
import socket
import random
import platform
import psutil
import uuid
from scapy.all import get_if_addr, get_if_list
import re
import subprocess
from pathlib import Path
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor
import traceback

DEBUG = int(os.getenv("DEBUG", default="0"))
DEBUG_DISCOVERY = int(os.getenv("DEBUG_DISCOVERY", default="0"))
VERSION = "0.0.1"

exo_text = r"""
  _____  _____  
 / _ \ \/ / _ \ 
|  __/>  < (_) |
 \___/_/\_\___/ 
    """

# Single shared thread pool for subprocess operations
subprocess_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="subprocess_worker")


def get_system_info():
  if psutil.MACOS:
    if platform.machine() == "arm64":
      return "Apple Silicon Mac"
    if platform.machine() in ["x86_64", "i386"]:
      return "Intel Mac"
    return "Unknown Mac architecture"
  if psutil.LINUX:
    return "Linux"
  return "Non-Mac, non-Linux system"


def find_available_port(host: str = "", min_port: int = 49152, max_port: int = 65535) -> int:
  used_ports_file = os.path.join(tempfile.gettempdir(), "exo_used_ports")

  def read_used_ports():
    if os.path.exists(used_ports_file):
      with open(used_ports_file, "r") as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]
    return []

  def write_used_port(port, used_ports):
    with open(used_ports_file, "w") as f:
      print(used_ports[-19:])
      for p in used_ports[-19:] + [port]:
        f.write(f"{p}\n")

  used_ports = read_used_ports()
  available_ports = set(range(min_port, max_port + 1)) - set(used_ports)

  while available_ports:
    port = random.choice(list(available_ports))
    if DEBUG >= 2: print(f"Trying to find available port {port=}")
    try:
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
      write_used_port(port, used_ports)
      return port
    except socket.error:
      available_ports.remove(port)

  raise RuntimeError("No available ports in the specified range")


def print_exo():
  print(exo_text)


def print_yellow_exo():
  yellow = "\033[93m"  # ANSI escape code for yellow
  reset = "\033[0m"  # ANSI escape code to reset color
  print(f"{yellow}{exo_text}{reset}")


def terminal_link(uri, label=None):
  if label is None:
    label = uri
  parameters = ""

  # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST
  escape_mask = "\033]8;{};{}\033\\{}\033]8;;\033\\"

  return escape_mask.format(parameters, uri, label)


T = TypeVar("T")
K = TypeVar("K")


class AsyncCallback(Generic[T]):
  def __init__(self) -> None:
    self.condition: asyncio.Condition = asyncio.Condition()
    self.result: Optional[Tuple[T, ...]] = None
    self.sync_observers: list[Callable[..., None]] = []
    self.async_observers: list[Callable[..., object]] = []  # Could return Awaitable[None] but that's not available in all Python versions

  async def wait(self, check_condition: Callable[..., object], timeout: Optional[float] = None) -> Tuple[T, ...]:
    """
    Wait for a condition to be met. The check_condition can be either a sync or async function.
    If it's async, it will be awaited properly.
    
    Args:
        check_condition: Function that takes the result values and returns a boolean (or awaitable boolean)
        timeout: Maximum time to wait in seconds
    
    Returns:
        The result tuple once the condition is met
    """
    async with self.condition:
      # Create a wrapper that handles both sync and async check conditions
      async def condition_wrapper():
        if self.result is None:
          return False
          
        result = check_condition(*self.result)
        # If it's awaitable, await it
        if asyncio.iscoroutine(result):
          return await result
        # Otherwise return directly
        return result
      
      try:
        # Use asyncio.wait_for directly on our wrapper with the condition notifier
        async def wait_for_condition():
          while True:
            if await condition_wrapper():
              return True
            await self.condition.wait()
            
        await asyncio.wait_for(wait_for_condition(), timeout)
        
        # If we reach here, the condition was met
        assert self.result is not None
        return self.result
      except asyncio.TimeoutError:
        # If we time out, raise a more descriptive error
        if self.result is None:
          raise asyncio.TimeoutError("Timed out waiting for result to be set")
        else:
          raise asyncio.TimeoutError("Timed out waiting for condition to be met")

  def on_next(self, callback: Callable[..., None]) -> None:
    """Register a synchronous callback function"""
    self.sync_observers.append(callback)
    
  def on_next_async(self, callback: Callable[..., object]) -> None:
    """Register an asynchronous callback coroutine function"""
    self.async_observers.append(callback)

  def set(self, *args: T) -> None:
    """Set the result and call all observers"""
    self.result = args
    
    # Call synchronous observers directly
    for observer in self.sync_observers:
      observer(*args)
      
    # Schedule async observers as tasks if there's a running event loop
    try:
      loop = asyncio.get_running_loop()
      for async_observer in self.async_observers:
        loop.create_task(async_observer(*args))
      
      # Notify any waiters
      loop.create_task(self.notify())
    except RuntimeError:
      # If there's no running event loop, we can't run async callbacks
      # This usually happens in tests or when used outside of async code
      if self.async_observers and DEBUG >= 1:
        print("Warning: Async callbacks skipped because there's no running event loop")

  async def notify(self) -> None:
    async with self.condition:
      self.condition.notify_all()


class AsyncCallbackSystem(Generic[K, T]):
  def __init__(self) -> None:
    self.callbacks: Dict[K, AsyncCallback[T]] = {}

  def register(self, name: K) -> AsyncCallback[T]:
    if name not in self.callbacks:
      self.callbacks[name] = AsyncCallback[T]()
    return self.callbacks[name]

  def deregister(self, name: K) -> None:
    if name in self.callbacks:
      del self.callbacks[name]

  def trigger(self, name: K, *args: T) -> None:
    if name in self.callbacks:
      self.callbacks[name].set(*args)

  def trigger_all(self, *args: T) -> None:
    for callback in self.callbacks.values():
      callback.set(*args)


K = TypeVar('K', bound=str)
V = TypeVar('V')


class PrefixDict(Generic[K, V]):
  def __init__(self):
    self.items: Dict[K, V] = {}

  def add(self, key: K, value: V) -> None:
    self.items[key] = value

  def find_prefix(self, argument: str) -> List[Tuple[K, V]]:
    return [(key, value) for key, value in self.items.items() if argument.startswith(key)]

  def find_longest_prefix(self, argument: str) -> Optional[Tuple[K, V]]:
    matches = self.find_prefix(argument)
    if len(matches) == 0:
      return None

    return max(matches, key=lambda x: len(x[0]))


def is_valid_uuid(val):
  try:
    uuid.UUID(str(val))
    return True
  except ValueError:
    return False


def get_or_create_node_id():
  NODE_ID_FILE = Path(tempfile.gettempdir())/".exo_node_id"
  try:
    if NODE_ID_FILE.is_file():
      with open(NODE_ID_FILE, "r") as f:
        stored_id = f.read().strip()
      if is_valid_uuid(stored_id):
        if DEBUG >= 2: print(f"Retrieved existing node ID: {stored_id}")
        return stored_id
      else:
        if DEBUG >= 2: print("Stored ID is not a valid UUID. Generating a new one.")

    new_id = str(uuid.uuid4())
    with open(NODE_ID_FILE, "w") as f:
      f.write(new_id)

    if DEBUG >= 2: print(f"Generated and stored new node ID: {new_id}")
    return new_id
  except IOError as e:
    if DEBUG >= 2: print(f"IO error creating node_id: {e}")
    return str(uuid.uuid4())
  except Exception as e:
    if DEBUG >= 2: print(f"Unexpected error creating node_id: {e}")
    return str(uuid.uuid4())


def pretty_print_bytes(size_in_bytes: int) -> str:
  if size_in_bytes < 1024:
    return f"{size_in_bytes} B"
  elif size_in_bytes < 1024**2:
    return f"{size_in_bytes / 1024:.2f} KB"
  elif size_in_bytes < 1024**3:
    return f"{size_in_bytes / (1024 ** 2):.2f} MB"
  elif size_in_bytes < 1024**4:
    return f"{size_in_bytes / (1024 ** 3):.2f} GB"
  else:
    return f"{size_in_bytes / (1024 ** 4):.2f} TB"


def pretty_print_bytes_per_second(bytes_per_second: int) -> str:
  if bytes_per_second < 1024:
    return f"{bytes_per_second} B/s"
  elif bytes_per_second < 1024**2:
    return f"{bytes_per_second / 1024:.2f} KB/s"
  elif bytes_per_second < 1024**3:
    return f"{bytes_per_second / (1024 ** 2):.2f} MB/s"
  elif bytes_per_second < 1024**4:
    return f"{bytes_per_second / (1024 ** 3):.2f} GB/s"
  else:
    return f"{bytes_per_second / (1024 ** 4):.2f} TB/s"


def get_all_ip_addresses_and_interfaces(include_ipv6: bool = True, filter_ipv6: bool = True):
    """
    Get all IP addresses and their associated interfaces.
    
    Args:
        include_ipv6: Whether to include IPv6 addresses
        filter_ipv6: Whether to filter out certain IPv6 addresses (link-local, ULA, etc.)
        
    Returns:
        List of (ip_address, interface_name) tuples
    """
    ip_addresses = []
    
    # Always add localhost to the list first
    ip_addresses.append(("localhost", "lo"))
    
    # Get IPv4 addresses using scapy
    for interface in get_if_list():
      try:
        ip = get_if_addr(interface)
        # Skip invalid/special IPs
        if ip.startswith("0.0.") or ip == "0.0.0.0" or ip == "127.0.0.1":
            continue
        simplified_interface = re.sub(r'^\\Device\\NPF_', '', interface)
        ip_addresses.append((ip, simplified_interface))
      except:
        if DEBUG >= 1: print(f"Failed to get IPv4 address for interface {interface}")
        if DEBUG >= 1: traceback.print_exc()
    
    # Get IPv6 addresses if requested
    if include_ipv6:
      try:
        import socket
        import psutil
        
        # Use psutil to get network interfaces and addresses
        for name, addrs in psutil.net_if_addrs().items():
          for addr in addrs:
            # Check if it's an IPv6 address (Address family 23 or 30 is IPv6)
            if addr.family in (socket.AF_INET6, 30):  # 30 is AF_INET6 on some systems
              address = addr.address.split('%')[0]  # Remove scope ID if present
              
              # Skip certain types of IPv6 addresses if filtering is enabled
              if filter_ipv6:
                # Skip link-local addresses (fe80::) 
                if address.startswith('fe80:'):
                  continue
                # Skip ULA addresses (fd00::/8)
                if address.startswith('fd'):
                  continue
                # Skip deprecated site-local addresses (fec0::/10)
                if address.startswith('fec') or address.startswith('fed') or address.startswith('fee') or address.startswith('fef'):
                  continue
                # Skip loopback addresses (::1)
                if address == "::1":
                  continue
                # Skip unspecified addresses (::)
                if address == "::":
                  continue
              
              # Add the IPv6 address and interface name
              simplified_interface = re.sub(r'^\\Device\\NPF_', '', name)
              ip_addresses.append((address, simplified_interface))
      except Exception as e:
        if DEBUG >= 1: print(f"Failed to get IPv6 addresses: {e}")
        if DEBUG >= 1: traceback.print_exc()
    
    # Remove duplicates but maintain order preference
    # (localhost first, then IPv4, then IPv6)
    seen = set()
    result = []
    for ip, iface in ip_addresses:
      if ip not in seen:
        seen.add(ip)
        result.append((ip, iface))
    
    return result



async def get_macos_interface_type(ifname: str) -> Optional[Tuple[int, str]]:
  try:
    # Use the shared subprocess_pool
    output = await asyncio.get_running_loop().run_in_executor(
      subprocess_pool, lambda: subprocess.run(['system_profiler', 'SPNetworkDataType', '-json'], capture_output=True, text=True, close_fds=True).stdout
    )

    data = json.loads(output)

    for interface in data.get('SPNetworkDataType', []):
      if interface.get('interface') == ifname:
        hardware = interface.get('hardware', '').lower()
        type_name = interface.get('type', '').lower()
        name = interface.get('_name', '').lower()

        if 'thunderbolt' in name:
          return (5, "Thunderbolt")
        if hardware == 'ethernet' or type_name == 'ethernet':
          if 'usb' in name:
            return (4, "Ethernet [USB]")
          return (4, "Ethernet")
        if hardware == 'airport' or type_name == 'airport' or 'wi-fi' in name:
          return (3, "WiFi")
        if type_name == 'vpn':
          return (1, "External Virtual")

  except Exception as e:
    if DEBUG >= 2: print(f"Error detecting macOS interface type: {e}")

  return None


async def get_interface_priority_and_type(ifname: str) -> Tuple[int, str]:
  # On macOS, try to get interface type using networksetup
  if psutil.MACOS:
    macos_type = await get_macos_interface_type(ifname)
    if macos_type is not None: return macos_type

  # Local container/virtual interfaces
  if (ifname.startswith(('docker', 'br-', 'veth', 'cni', 'flannel', 'calico', 'weave')) or 'bridge' in ifname):
    return (7, "Container Virtual")

  # Loopback interface
  if ifname.startswith('lo'):
    return (6, "Loopback")

  # Traditional detection for non-macOS systems or fallback
  if ifname.startswith(('tb', 'nx', 'ten')):
    return (5, "Thunderbolt")

  # Regular ethernet detection
  if ifname.startswith(('eth', 'en')) and not ifname.startswith(('en1', 'en0')):
    return (4, "Ethernet")

  # WiFi detection
  if ifname.startswith(('wlan', 'wifi', 'wl')) or ifname in ['en0', 'en1']:
    return (3, "WiFi")

  # Non-local virtual interfaces (VPNs, tunnels)
  if ifname.startswith(('tun', 'tap', 'vtun', 'utun', 'gif', 'stf', 'awdl', 'llw')):
    return (1, "External Virtual")

  # Other physical interfaces
  return (2, "Other")


async def shutdown(signal, loop, server):
  """Gracefully shutdown the server and close the asyncio loop."""
  print(f"Received exit signal {signal.name}...")
  print("Thank you for using exo.")
  print_yellow_exo()
  server_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
  [task.cancel() for task in server_tasks]
  print(f"Cancelling {len(server_tasks)} outstanding tasks")
  await asyncio.gather(*server_tasks, return_exceptions=True)
  await server.stop()


def is_frozen():
  return getattr(sys, 'frozen', False) or os.path.basename(sys.executable) == "exo" \
    or ('Contents/MacOS' in str(os.path.dirname(sys.executable))) \
    or '__nuitka__' in globals() or getattr(sys, '__compiled__', False)

async def get_mac_system_info() -> Tuple[str, str, int]:
    """Get Mac system information using system_profiler."""
    try:
        output = await asyncio.get_running_loop().run_in_executor(
            subprocess_pool,
            lambda: subprocess.check_output(["system_profiler", "SPHardwareDataType"]).decode("utf-8")
        )
        
        model_line = next((line for line in output.split("\n") if "Model Name" in line), None)
        model_id = model_line.split(": ")[1] if model_line else "Unknown Model"
        
        chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
        chip_id = chip_line.split(": ")[1] if chip_line else "Unknown Chip"
        
        memory_line = next((line for line in output.split("\n") if "Memory" in line), None)
        memory_str = memory_line.split(": ")[1] if memory_line else "Unknown Memory"
        memory_units = memory_str.split()
        memory_value = int(memory_units[0])
        memory = memory_value * 1024 if memory_units[1] == "GB" else memory_value
        
        return model_id, chip_id, memory
    except Exception as e:
        if DEBUG >= 2: print(f"Error getting Mac system info: {e}")
        return "Unknown Model", "Unknown Chip", 0

def get_exo_home() -> Path:
  if psutil.WINDOWS: docs_folder = Path(os.environ["USERPROFILE"])/"Documents"
  else: docs_folder = Path.home()/"Documents"
  if not docs_folder.exists(): docs_folder.mkdir(exist_ok=True)
  exo_folder = docs_folder/"Exo"
  if not exo_folder.exists(): exo_folder.mkdir(exist_ok=True)
  return exo_folder


def get_exo_config_dir() -> Path:
  config_dir = Path.home()/".exo"
  if not config_dir.exists(): config_dir.mkdir(exist_ok=True)
  return config_dir


def get_exo_images_dir() -> Path:
  exo_home = get_exo_home()
  images_dir = exo_home/"Images"
  if not images_dir.exists(): images_dir.mkdir(exist_ok=True)
  return images_dir
