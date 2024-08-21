import os
from pathlib import Path
import shutil
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm

label_dict = {0: 'Mug', 1: 'Keyboard', 2: 'Trainers', 3: 'ZombieThor', 4: 'Mouse', 5: 'Fan', 6: 'Monitor',
7: 'Keys', 8: 'Drawer', 9: 'Sunglasses', 10: 'PlasticBottle'}

colors = ['red', 'green', 'blue', 'purple', 'orange', 'pink', 'yellow', 'cyan', 'magenta', 'lime', 'brown']

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


def filter_files(first_direc, second_direc, first_ext, second_ext, destination_direc):
    """

    :param first_direc:
    :param second_direc:
    :param first_ext:
    :param second_ext:
    :param destination_direc:
    :return:
    """
    os.makedirs(destination_direc, exist_ok=True)
    first_list_files = [f for f in os.listdir(first_direc) if f.endswith(first_ext)]
    second_list_files = [f for f in os.listdir(second_direc) if f.endswith(second_ext)]
    not_present_files_count = 0
    present_files_count = 0
    for file in first_list_files:
        file_name = file.split(first_ext)[0]
        if file_name + second_ext in second_list_files:
            destination_file = os.path.join(destination_direc, file)
            shutil.copy(os.path.join(first_direc, file), destination_file)
            print(f"File {file} copied to {destination_file}")
            present_files_count += 1
        else:
            not_present_files_count += 1
    print(f"Total files copied: {present_files_count}")



def yolo_tobbox(img_width, img_height, yolo_box):
    x_center, y_center, width, height = yolo_box
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    return x1, y1, x2, y2


# Function to plot bounding boxes and labels on the image
def plot_bboxes(image_path, bboxes, labels):
    # Load the image
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        # Load a larger font. Adjust path and size as necessary.
        try:
            font = ImageFont.truetype("arial.ttf", 24)  # Larger font size
        except IOError:
            font = ImageFont.load_default()

        # Draw each bounding box and label with a different color
        for bbox, label, color in zip(bboxes, labels, colors):
            draw.rectangle(bbox, outline=color, width=3)
            draw.text((bbox[0], bbox[1] - 30), label, fill=color, font=font)
        # Display the image
        plt.imshow(img)
        plt.axis('off')
        plt.show()


# Main function to process the files
def process_files(txt_folder, img_folder):
    for filename in os.listdir(txt_folder):
        if filename.endswith('.txt'):
            image_path = os.path.join(img_folder, filename.replace('.txt', '.jpg'))
            txt_path = os.path.join(txt_folder, filename)

            if os.path.exists(image_path):
                # Read the bounding boxes from txt file
                with open(txt_path, 'r') as file:
                    bboxes = []
                    labels = []
                    for line in file.readlines():
                        parts = line.strip().split()
                        class_id, x_center, y_center, width, height = map(float, parts)
                        with Image.open(image_path) as img:
                            img_width, img_height = img.size
                            bbox = yolo_tobbox(img_width, img_height, (x_center, y_center, width, height))
                            bboxes.append(bbox)
                            labels.append(label_dict[int(class_id)])
                # Plot bounding boxes on the image
                plot_bboxes(image_path, bboxes, labels)


def resize_images(source_directory, target_directory, target_size):
    """
    Resizes all images in the specified source directory to the target size and saves them to a new directory,
    including a progress bar for the operation.

    Args:
    source_directory (str): The path to the directory containing the images to be resized.
    target_directory (str): The path to the directory where resized images will be saved.
    target_size (tuple): The desired size of the images as (width, height).
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)

    # List all image files
    files = [f for f in os.listdir(source_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Loop through all files in the directory with a progress bar
    for filename in tqdm(files, desc="Resizing images"):
        source_path = os.path.join(source_directory, filename)
        target_path = os.path.join(target_directory, filename)
        with Image.open(source_path) as img:
            # Resize the image
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)  # Updated for newer versions of Pillow
            # Save the resized image to new directory
            resized_img.save(target_path)




if __name__ == '__main__':
    FIRST_DIREC, FIRST_EXT = r'C:\Users\sadebayo\Downloads\Photos-001', '.jpg'
    SECOND_DIREC, SECOND_EXT = r'C:\Users\sadebayo\Downloads\labels_my-project-name_2024-08-21-07-18-36', '.txt'
    # DESTINATION_DIREC = 'images'
    # filter_files(FIRST_DIREC, SECOND_DIREC, FIRST_EXT, SECOND_EXT, DESTINATION_DIREC)
    text_folder, image_folder = 'data/labels', 'data/resized_images'
    # resize_images(image_folder, 'data/resized_images-', (640, 480))
    process_files(text_folder, image_folder)

