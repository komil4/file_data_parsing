import os
import cv2
import numpy as np
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from pdf2image import convert_from_path


# -------------------------------------
# Работа с диском
# Save image to disk
def save_image_to_disk(path, filename, image):
    if os.path.exists(path + filename):
        delete_file(path, filename)

    is_success, im_buf_arr = cv2.imencode(".jpeg", image)
    im_buf_arr.tofile(path + filename)


# Read image from disk
def read_image_from_disk(path, filename):
    stream = open(path + filename, 'rb')
    bytes = bytearray(stream.read())
    array = np.asarray(bytes, dtype=np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    return img


# Save files to disk
# If file type PDF to filepath, if Image to imagepath
def save_files_to_disk(path, files):
    file_names = []
    for file_name in files:
        file = files.get(file_name)
        file.save(path + file.filename)
        file.stream.seek(0)
        file_names.append(file.filename)

    return file_names


# Запаковка файлов
def path_files_to_zip(path, files):
    zip_file_name = str(uuid.uuid4()) + '.zip'
    zip_file_name_with_path = path + zip_file_name

    z = zipfile.ZipFile(zip_file_name_with_path, 'w')
    for file in files:
        z.write(path + file, file)
    z.close()

    return zip_file_name


# Запаковка изображний
def path_images_to_zip(path, images):
    image_filenames = []
    for image in images:
        image_filenames.append(image.get("filename"))

    zip_file_name = path_files_to_zip(path, image_filenames)

    return zip_file_name


# Delete file
def delete_file(path, filename):
    try:
        os.remove(path + filename)
    except:
        print("Cannot remove file")


# -------------------------------------
# Базовые функции работы с файлами и изображениями
# Split PDF file to few images
def split_pdf_doc(path, filename, file_suffix='.jpg', image_type='JPEG'):
    file_names = []
    print("Start convert PDF file " + filename + " to JPEG")

    pages = convert_from_path(path + filename, thread_count=4)
    for i in range(len(pages)):
        page_filename = get_filename_without_type(filename) + '_page_' + str(i) + file_suffix
        pages[i].save(path + page_filename, image_type)
        file_names.append(page_filename)

    print("Convert success")

    return file_names


# Get images list from files
# If the file on PDF format he has few pages
# Remove files fromfilepath and create files on image path
def get_images_from_files(path, file_names):
    images = []

    # Working with files
    for filename in file_names:
        # If file in PDF format
        if get_file_type(filename) == ".pdf":
            # Split PDF to images
            files = split_pdf_doc(path, filename)
            for image_file in files:
                image = {'path': path, 'name': filename, 'image': read_image_from_disk(path, image_file)}
                images.append(image)
        else:
            image = {'path': path, 'name': filename, 'image': read_image_from_disk(path, filename)}
            images.append(image)

    return images


# -------------------------------------
# More functions
# Get filename without type
def get_filename_without_type(filename):
    name = PurePosixPath(filename).stem
    return name


# Get file type
def get_file_type(filename):
    suffix = PurePosixPath(filename).suffix
    return suffix


# Функция возвращает новое имя для папки
# Имя папки должно быть уникально
def get_new_path_name(path):
    name = uuid.uuid4().hex
    while os.path.exists(path + name):
        name = uuid.uuid4().hex

    return name
