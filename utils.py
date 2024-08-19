import os
from pathlib import Path

def list_file_extensions(directory):
    # Set to hold unique extensions
    extensions = set()
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Use pathlib to extract file extensions
            extension = Path(file).suffix
            # Add the extension to the set
            extensions.add(extension)
    # Return all unique extensions
    return extensions


def list_files(directory, prefix='', limit=None):
    # Check if the directory exists and is accessible
    if not os.path.exists(directory) or not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory or cannot be accessed.")
        return

    # List files in the specified directory
    try:
        files = os.listdir(directory)
    except PermissionError:
        print(f"Cannot access contents of {directory}")
        return

    # Sort files by name
    files.sort()

    # Setup a counter to limit the number of files displayed
    count = 0

    # Iterate through the files and directories
    for file in files:
        if limit is not None and count >= limit:
            break
        path = os.path.join(directory, file)
        if os.path.isdir(path):
            # It's a directory, print and recurse
            print(f"{prefix}├── {file}/")  # Indicate it's a directory with '/'
            # Recursive call, increasing the prefix to indicate depth
            list_files(path, prefix + '│   ', limit)
        else:
            # It's a file, just print
            print(f"{prefix}├── {file}")
        count += 1
