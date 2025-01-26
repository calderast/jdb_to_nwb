"""
This script downloads data from the user's UCSF Box folder "nwb-test-data" to the tests/test_data directory
using the Box credentials in the .env file. By default, existing files are not overwritten unless the --overwrite flag
is used. Downloads are parallelized across multiple workers.
"""

import argparse
from ftplib import FTP_TLS
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, current_thread
from tqdm.auto import tqdm
import multiprocessing
import threading

# List of file extensions to exclude from the download
EXCLUDE_FILES = [".dat"]

# Lock for thread-safe printing
print_lock = Lock()

# Add thread-local storage for progress bar positions
thread_data = threading.local()
position_lock = Lock()
next_position = 0
active_positions = set()  # Track which positions are in use

def safe_print(*args, **kwargs):
    """Thread-safe printing function"""
    with print_lock:
        print(*args, **kwargs)

def human_readable_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return "Unknown"

def create_ftps_connection():
    """Create and return a new FTP_TLS connection"""
    box_username = os.environ.get("BOX_USERNAME")
    box_password = os.environ.get("BOX_PASSWORD")
    ftp_host = "ftp.box.com"
    ftps = FTP_TLS(host=ftp_host, user=box_username, passwd=box_password)
    ftps.prot_p()
    return ftps

def get_thread_number():
    """Extract just the thread number from the thread name"""
    thread_name = current_thread().name
    # ThreadPoolExecutor-0_2 -> 2
    try:
        return int(thread_name.split('_')[-1])
    except (ValueError, IndexError):
        return 0

def create_padded_description(thread_num: int, filename: str, width: int = 100) -> str:
    """Create a left-aligned padded description for the progress bar"""
    if thread_num == -1:  # Special case for total progress
        desc = "Total Progress"
    else:
        desc = f"[Thread {thread_num}] {filename}"
    return f"{desc:<{width}}"

def get_thread_position():
    """Get or create a fixed position for the current thread"""
    global next_position
    if not hasattr(thread_data, 'position'):
        with position_lock:
            # Find the first available position, starting at position 1
            # (position 0 is reserved for overall progress)
            pos = 1
            while pos in active_positions:
                pos += 1
            thread_data.position = pos
            active_positions.add(pos)
    return thread_data.position

def release_thread_position():
    """Release the thread's position when done"""
    if hasattr(thread_data, 'position'):
        with position_lock:
            active_positions.remove(thread_data.position)
            del thread_data.position

def download_file(remote_file: str, local_file: Path, size_bytes: int):
    """Download a single file using its own FTP connection"""
    thread_num = get_thread_number()
    filename = os.path.basename(remote_file)
    position = get_thread_position()
    
    # Create a new FTP connection for this thread
    ftps = create_ftps_connection()
    try:
        # Navigate to the correct directory
        remote_dir = os.path.dirname(remote_file)
        if remote_dir:
            ftps.cwd(remote_dir)
        
        # Create progress bar with padded description
        with tqdm(
            total=size_bytes,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=create_padded_description(thread_num, filename),
            leave=False,
            position=position,  # Will start at position 1 or higher
            bar_format='{desc} {percentage:3.0f}%|{bar:10}{r_bar}'
        ) as pbar:
            # Create a callback to update the progress bar
            def callback(data):
                pbar.update(len(data))
                local_file_obj.write(data)
                
            with open(local_file, "wb") as local_file_obj:
                ftps.retrbinary(f"RETR {os.path.basename(remote_file)}", callback)
    finally:
        ftps.quit()
        release_thread_position()  # Release the position when done

def collect_files_to_download(ftps: FTP_TLS, remote_dir: str, local_base_dir: Path, args) -> list:
    """Collect all files to download from a directory"""
    files_to_download = []
    
    # Save and restore directory
    original_dir = ftps.pwd()
    try:
        ftps.cwd(remote_dir)
        current_dir = ftps.pwd()
        
        # Create corresponding local directory
        # Extract the path after the base remote directory
        relative_path = Path(remote_dir.split(f"{args.base_remote_dir}/")[-1])
        local_dir = local_base_dir / relative_path
        local_dir.mkdir(parents=True, exist_ok=True)

        # List all entries in the directory
        for entry in ftps.mlsd():
            name, facts = entry

            if facts["type"] == "dir" and name not in [".", ".."]:
                # Recursively get files from subdirectory
                subdir_path = f"{current_dir}/{name}"
                subdir_files = collect_files_to_download(ftps, subdir_path, local_dir, args)
                files_to_download.extend(subdir_files)
                
            elif facts["type"] == "file":
                remote_file_path = f"{current_dir}/{name}"
                local_file_path = local_dir / name

                # Skip files with excluded extensions
                if any(name.endswith(ext) for ext in EXCLUDE_FILES):
                    safe_print(f"Skipping {remote_file_path} because it has an excluded extension")
                    continue

                # Skip if file exists and overwrite flag not set
                if local_file_path.exists() and not args.overwrite:
                    safe_print(f"Skipping {remote_file_path} because it already exists")
                    continue

                size_bytes = ftps.size(name)
                files_to_download.append((remote_file_path, local_file_path, size_bytes))
                
    finally:
        ftps.cwd(original_dir)
        
    return files_to_download

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--workers", 
        type=int, 
        default=-1,
        help="Number of parallel download workers. Use -1 to use all available CPU cores (default: -1)"
    )
    parser.add_argument(
        "--base-remote-dir",
        type=str,
        default="nwb-test-data",
        help=(
            "Base remote directory in the user's Box account which remote paths to data are relative to "
            "(default: nwb-test-data)"
        )
    )
    args = parser.parse_args()

    # These paths are relative to the base remote directory
    remote_dirs_to_download = [
        "IM-1478/07252022",
        "IM-1770_corvette/11062024",
    ]

    # Prepend the base remote directory to each path
    full_remote_dirs = [f"/{args.base_remote_dir}/{path}" for path in remote_dirs_to_download]

    # Base directory for downloads
    base_dir = Path("tests/test_data/downloaded")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Read .env file
    env_file_path = ".env"
    if os.path.exists(env_file_path):
        with open(env_file_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

    # Initialize FTP connection for listing directories
    ftps = create_ftps_connection()
    
    try:
        # Collect all files to download
        all_files = []
        for remote_dir in full_remote_dirs:
            safe_print(f"Collecting files from {remote_dir}...")
            files = collect_files_to_download(ftps, remote_dir, base_dir, args)
            all_files.extend(files)
            
        if not all_files:
            safe_print("No files to download")
            return
            
        # Calculate total size
        total_size = sum(size for _, _, size in all_files)
        safe_print(f"Found {len(all_files)} files to download ({human_readable_size(total_size)})")
        
        # Set number of workers
        max_workers = multiprocessing.cpu_count() if args.workers < 1 else args.workers
        
        # Create progress bar for overall progress
        with tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=create_padded_description(-1, "", width=20),  # Use -1 to indicate total progress
            position=0,
            leave=True,
            bar_format='{desc} {percentage:3.0f}%|{bar:90}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ) as total_pbar:
            # Download all files in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for remote_file, local_file, size_bytes in all_files:
                    future = executor.submit(
                        download_file, remote_file, local_file, size_bytes
                    )
                    # Add callback to update total progress
                    future.add_done_callback(
                        lambda _, size=size_bytes: total_pbar.update(size)
                    )
                    futures.append(future)

                # Wait for all downloads to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        safe_print(f"Error during download: {str(e)}")
    finally:
        ftps.quit()

if __name__ == "__main__":
    main()
