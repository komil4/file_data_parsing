import os
import argparse
import cv2
import logging

from flask import Flask, request, json, render_template, url_for, send_from_directory
from matplotlib import pyplot as plt

import imageOperation
import filesystem
import image
import imageStructure


#fronend
#PyMuPDF


# -------------------------------------
# Вывод изображений
# Show image
def show_image(image, name="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(name)
    plt.show()


# Show image in full screen
def show_image_full_screen(image, name="Image"):
    cv2.imshow(name, image)
    cv2.waitKey(0)


# -------------------------------------
# Arguments CLI
parser = argparse.ArgumentParser(description='Set the server arguments')
parser.add_argument("-host", default='127.0.0.1', help='This is a hostname value. If you want to start in docker, set 0.0.0.0')
parser.add_argument("-port", default='3001', help='This is a port value. Default set 3000')
args = parser.parse_args()


# Global values
HTTP_HOSTNAME = args.host
HTTP_PORT = args.port
TEMP_PATH = (os.getcwd() + '/temp\\').replace('\\', '/')


# -------------------------------------
# Logging settings
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


# -------------------------------------
# App start
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'results'


# -------------------------------------
# Модуль работы с HTTP сервисом
# HTTP service methods
# Parsing PDF to Jpeg
@app.route("/getJpegImagesFromFiles", methods=["POST"])
def get_jpeg_images_from_files():
    if request.args.get('color') == "1":
        color = True
    else:
        color = False
    if request.args.get('threshold') == "1":
        threshold = True
    else:
        threshold = False
    if request.args.get('autorotate') == "1":
        autorotate = True
    else:
        autorotate = True

    # Новая папка для временного хранения файлов
    # Это же значение пусть будет дальше УИДом сессии
    pathname = filesystem.get_new_path_name(TEMP_PATH)
    path = TEMP_PATH + pathname + "/"

    # Получили массив изображений
    file_names = filesystem.save_files_to_disk(path, request.files, pathname)
    images = filesystem.get_images_from_files(path, file_names, pathname)

    image.images_process(images,
            [{'name': 'autorotate_image',      'parameters': ['', '']},
             {'name': 'gray_scale',            'parameters': ['', '']},
             # {'name': 'save_image',            'parameters': ['gray_scale', True]},
             {'name': 'gaussian_threshold',    'parameters': ['', '']},
             # {'name': 'save_image',            'parameters': ['gaussian_threshold', True]},
             {'name': 'vertical_lines',        'parameters': ['gaussian_threshold', 'vertical_lines_negative']},
             {'name': 'save_image',            'parameters': ['vertical_lines_negative', True]},
             {'name': 'horizontal_lines',      'parameters': ['gaussian_threshold', 'horizontal_lines_negative']},
             {'name': 'save_image',            'parameters': ['horizontal_lines_negative', True]},
             {'name': 'table_tines',           'parameters': [['horizontal_lines_negative', 'vertical_lines_negative'], 'table_tines_negative']},
             {'name': 'save_image'}
             ])

    for img in images:
        lines = img.get_image_by_step_name()
        polygon = imageStructure.Polygon(lines, img.name, pathname)
        polygon.group_all_blocks()
        rotate_image = img.get_image_by_step_name('autorotate')
        if len(rotate_image.shape) == 2:
            rotate_image = cv2.cvtColor(rotate_image, cv2.COLOR_GRAY2RGB)
        polygon.draw_boxes_and_save(path, rotate_image)

    '''
    if autorotate:
        imageOperation.auto_rotate_images(images)

    if threshold:
        imageOperation.threshold_images(images)
    elif color:
        imageOperation.convert_to_grayscale_images(images)

    if True:
        imageOperation.delete_trash_tables_from_images(images, True)
        '''

    zip_file_name = filesystem.path_images_to_zip(path, images, pathname)

    response = send_from_directory(path, zip_file_name)
    # delete_file(zip_file_name)

    return response


@app.route("/getRotateImages/<degree>", methods=["POST"])
def get_rotate_images(degree):
    pathname = filesystem.get_new_path_name(TEMP_PATH)
    path = TEMP_PATH + pathname + "/"

    file_names = filesystem.save_files_to_disk(path, request.files)
    images = filesystem.get_images_from_files(path, file_names)

    imageOperation.rotate_images(images, degree)

    zip_file_name = filesystem.path_images_to_zip(path, images)

    response = send_from_directory(path, zip_file_name)

    return response


@app.route("/")
@app.route("/index")
def home():
    return render_template("index.html", processing=False)


if __name__ == "__main__":
    app.run(host=HTTP_HOSTNAME, port=HTTP_PORT, debug=True, use_reloader=False)
