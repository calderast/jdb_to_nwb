"""
This script downloads data from the user's UCSF Box folder "nwb-test-data" to the tests/test_data directory 
using the Box credentials in the .env file. Existing files in the test_data directory are overwritten.
"""

from ftplib import FTP_TLS
import os
from pathlib import Path


def main():
    # This path is relative to the root of the user's Box account
    remote_dir_to_download = "/nwb-test-data/IM-1478/07252022"

    # List of file extensions to exclude from the download
    exclude_files = [".dat"]

    # This is the directory where the downloaded data will be saved
    test_data_dir = Path('tests/test_data/downloaded/IM-1478/07252022')

    # Create test data directory if it doesn't exist
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Read .env file and parse variables into environment variables 
    with open('.env', 'r') as f:
        for line in f:
            # Skip empty lines and comments
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

    # Get Box credentials
    box_username = os.environ.get('BOX_USERNAME')
    box_password = os.environ.get('BOX_PASSWORD')

    # Initialize FTP over TLS connection
    ftp_host = "ftp.box.com"
    ftps = FTP_TLS(host=ftp_host, user=box_username, passwd=box_password)
    ftps.prot_p()

    def human_readable_size(size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return "Unknown"

    # List and download all files in the directory recursively
    def download_recursively(ftps: FTP_TLS, remote_dir: str, local_dir: Path):
        # Save current directory to restore later
        original_dir = ftps.pwd()
        
        # Update current path
        ftps.cwd(remote_dir)
        current_dir = ftps.pwd()
        
        # Create local directory if it doesn't exist
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # List directory contents with metadata
        for entry in ftps.mlsd():
            name, facts = entry
            
            if facts['type'] == 'dir':
                # Recursively download subdirectories (skip "." and "..")
                if name not in [".", ".."]:
                    download_recursively(ftps, name, local_dir / name)
            elif facts['type'] == 'file':
                # Download file
                remote_file_path = current_dir + "/" + name

                # Skip files with excluded extensions
                if any(name.endswith(ext) for ext in exclude_files):
                    print(f"Skipping {remote_file_path} because it has an excluded extension")
                    continue

                local_file_path = local_dir / name
                size_bytes = ftps.size(name)
                size_str = human_readable_size(size_bytes)
                with open(local_file_path, 'wb') as local_file:
                    print(f"Downloading {remote_file_path} ({size_str})... ", end="", flush=True)
                    ftps.retrbinary(f'RETR {name}', local_file.write)
                    print("done")
            else:
                print('Unknown type: ' + facts['type'])
        
        # Return to original directory
        ftps.cwd(original_dir)

    # Start recursive download from current directory
    print(f"Starting recursive download from {remote_dir_to_download} to {test_data_dir}...")
    download_recursively(ftps, remote_dir_to_download, test_data_dir)

    # Close the connection
    ftps.quit()

if __name__ == "__main__":
    main()