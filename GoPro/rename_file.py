import os
import time
from PIL import Image
from PIL.ExifTags import TAGS
import glob

def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    if exif_data:
        exif = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
        return exif
    else:
        return None

def get_photo_datetime(exif_data):
    # 'DateTimeOriginal' contains the date and time the photo was taken
    date_time_str = exif_data.get('DateTimeOriginal', None)
    if date_time_str:
        # Convert the string "YYYY:MM:DD HH:MM:SS" to "YYYY-MM-DD HH:MM:SS"
        date_time_str = date_time_str.replace(':', '-', 2)
        return date_time_str
    else:
        return None

def rename_to_epoch(image_path):
    exif_data = get_exif_data(image_path)
    if not exif_data:
        print(f"No EXIF metadata found for {image_path}")
        return

    date_time_str = get_photo_datetime(exif_data)
    if not date_time_str:
        print(f"Date and time not found in metadata for {image_path}")
        return
    
    # Convert the date and time to a Unix timestamp (epoch time)
    epoch_time = int(time.mktime(time.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")))
    
    # Get the directory and file extension
    directory, original_filename = os.path.split(image_path)
    file_extension = os.path.splitext(original_filename)[1]

    # New filename: epoch time + original extension
    new_filename = f"{epoch_time}{file_extension}"
    new_file_path = os.path.join(directory, new_filename)

    # Rename the file
    os.rename(image_path, new_file_path)
    print(f"Renamed '{image_path}' to '{new_file_path}'")

def process_folder(folder_path):
    # Search for all image files in the folder with common image extensions
    image_files = glob.glob(os.path.join(folder_path, "*.[jJpP][pPnN][gG]"))  # This includes .jpg, .png, .jpeg

    # Iterate over each image file and rename it based on EXIF metadata
    for image_path in image_files:
        rename_to_epoch(image_path)

# Example usage
folder_path = r'GoPro\100GOPRO'
process_folder(folder_path)
