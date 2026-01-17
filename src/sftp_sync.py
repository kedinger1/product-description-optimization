"""
SFTP Feed Sync - Downloads product feed from Feedonomics SFTP
Designed to run as a Render cron job
"""

import os
import paramiko
from pathlib import Path
from datetime import datetime

# SFTP Configuration (from environment variables)
SFTP_HOST = os.environ.get('SFTP_HOST', 'sftpgo.feedonomics.com')
SFTP_USERNAME = os.environ.get('SFTP_USERNAME')
SFTP_PASSWORD = os.environ.get('SFTP_PASSWORD')
SFTP_REMOTE_PATH = os.environ.get('SFTP_REMOTE_PATH', '/incoming/src/feedonomics/catalog')
SFTP_FILENAME = os.environ.get('SFTP_FILENAME', 'google_products_ns-company_en_US.csv')

# Local paths
BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / 'uploads'


def download_feed():
    """Download the product feed from SFTP"""

    if not SFTP_USERNAME or not SFTP_PASSWORD:
        raise ValueError("SFTP_USERNAME and SFTP_PASSWORD environment variables are required")

    UPLOADS_DIR.mkdir(exist_ok=True)

    print(f"[{datetime.now().isoformat()}] Starting SFTP sync...")
    print(f"Host: {SFTP_HOST}")
    print(f"Remote path: {SFTP_REMOTE_PATH}/{SFTP_FILENAME}")

    # Connect to SFTP
    transport = paramiko.Transport((SFTP_HOST, 22))
    transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)

    try:
        remote_file = f"{SFTP_REMOTE_PATH}/{SFTP_FILENAME}"
        local_file = UPLOADS_DIR / SFTP_FILENAME

        # Get remote file info
        file_stat = sftp.stat(remote_file)
        remote_size = file_stat.st_size
        remote_mtime = datetime.fromtimestamp(file_stat.st_mtime)

        print(f"Remote file size: {remote_size / 1024 / 1024:.2f} MB")
        print(f"Remote file modified: {remote_mtime.isoformat()}")

        # Download the file
        print(f"Downloading to {local_file}...")
        sftp.get(remote_file, str(local_file))

        # Verify download
        local_size = local_file.stat().st_size
        print(f"Downloaded: {local_size / 1024 / 1024:.2f} MB")

        if local_size == remote_size:
            print("Download verified successfully!")
        else:
            print(f"WARNING: Size mismatch! Remote: {remote_size}, Local: {local_size}")

    finally:
        sftp.close()
        transport.close()

    print(f"[{datetime.now().isoformat()}] SFTP sync complete!")
    return str(local_file)


if __name__ == '__main__':
    download_feed()
