import os
import argparse
import cv2
import logging

from flask import Flask, request, json, render_template, url_for, send_from_directory
from matplotlib import pyplot as plt

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
@app.route("/getJpegImagesFromFiles_l", methods=["POST"])
def get_jpeg_images_from_files():
    actions = []
    if request.args.get('autorotate') == "1":
        actions.append({'name': 'autorotate_image', 'parameters': ['main', 'autorotate_image']})
    if request.args.get('color') == "1":
        actions.append({'name': 'gray_scale', 'parameters': ['autorotate_image', 'gray_scale']})
        if request.args.get('threshold') == "1":
            actions.append({'name': 'gaussian_threshold', 'parameters': ['gray_scale', '']})

    # Новая папка для временного хранения файлов
    # Это же значение пусть будет дальше УИДом сессии
    pathname = filesystem.get_new_path_name(TEMP_PATH)
    path = TEMP_PATH + pathname + "/"

    # Получили массив изображений
    file_names = filesystem.save_files_to_disk(path, request.files, pathname)
    images = filesystem.get_images_from_files(path, file_names, pathname)

    actions.append({'name': 'save_image',            'parameters': ['', False]})

    image.images_process(images, actions)

    zip_file_name = filesystem.path_images_to_zip(path, images, pathname)

    response = send_from_directory(path, zip_file_name)
    # delete_file(zip_file_name)

    return response


@app.route("/getRotateImages/<degree>", methods=["POST"])
def get_rotate_images(degree):
    actions = [{'name': 'rotate_image', 'parameters': [['main', degree], 'rotate_image']}]

    # Новая папка для временного хранения файлов
    # Это же значение пусть будет дальше УИДом сессии
    pathname = filesystem.get_new_path_name(TEMP_PATH)
    path = TEMP_PATH + pathname + "/"

    # Получили массив изображений
    file_names = filesystem.save_files_to_disk(path, request.files, pathname)
    images = filesystem.get_images_from_files(path, file_names, pathname)

    actions.append({'name': 'save_image', 'parameters': ['', False]})

    image.images_process(images, actions)

    zip_file_name = filesystem.path_images_to_zip(path, images, pathname)

    response = send_from_directory(path, zip_file_name)
    # delete_file(zip_file_name)

    return response


@app.route("/getJpegImagesFromFiles", methods=["POST"])
def get_cutted_images():
    # Новая папка для временного хранения файлов
    # Это же значение пусть будет дальше УИДом сессии
    pathname = filesystem.get_new_path_name(TEMP_PATH)
    path = TEMP_PATH + pathname + "/"

    # Получили массив изображений
    file_names = filesystem.save_files_to_disk(path, request.files, pathname)
    images = filesystem.get_images_from_files(path, file_names, pathname)

    image.images_process(images,
                         [{'name': 'autorotate_image', 'parameters': ['main', 'autorotate_image']},
                          {'name': 'gray_scale', 'parameters': ['autorotate_image', 'gray_scale']},
                          # {'name': 'save_image',            'parameters': ['gray_scale', True]},
                          {'name': 'gaussian_threshold', 'parameters': ['gray_scale', '']},
                          # {'name': 'save_image',            'parameters': ['gaussian_threshold', True]},
                          {'name': 'vertical_lines', 'parameters': ['gaussian_threshold', 'vertical_lines_negative']},
                          {'name': 'horizontal_lines',
                           'parameters': ['gaussian_threshold', 'horizontal_lines_negative']},
                          {'name': 'table_tines',
                           'parameters': [['horizontal_lines_negative', 'vertical_lines_negative'],
                                          'table_tines_negative']},
                          # {'name': 'save_image',            'parameters': ['vertical_lines_negative', True]},
                          {'name': 'align_table',
                           'parameters': [['vertical_lines_negative', 'gray_scale'], 'align_table']},
                          {'name': 'align_table',
                           'parameters': [['vertical_lines_negative', 'table_tines_negative'], 'align_table_tines_negative']},
                          {'name': 'save_image', 'parameters': ['align_table', True]},
                          {'name': 'save_image', 'parameters': ['align_table_tines_negative', True]},
                          # {'name': 'gaussian_threshold', 'parameters': ['align_table', 'gaussian_threshold']},
                          # {'name': 'save_image', 'parameters': ['gaussian_threshold', True]},
                          # {'name': 'vertical_lines', 'parameters': ['gaussian_threshold', 'vertical_lines_negative']},
                          # {'name': 'save_image',            'parameters': ['vertical_lines_negative', True]},
                          # {'name': 'horizontal_lines',
                            # 'parameters': ['gaussian_threshold', 'horizontal_lines_negative']},
                          # {'name': 'save_image',            'parameters': ['horizontal_lines_negative', True]},
                          # {'name': 'table_tines',
                           # 'parameters': [['horizontal_lines_negative', 'vertical_lines_negative'],
                          #                'table_tines_negative']},
                          # {'name': 'save_image', 'parameters': ['table_tines', True]}
                          ])
    for img in images:
        lines = img.get_image_by_step_name('align_table_tines_negative')
        polygon = imageStructure.Polygon(lines, img.name, pathname)
        rotate_image = img.get_image_by_step_name('align_table')

        if len(rotate_image.shape) == 2:
            rotate_image = cv2.cvtColor(rotate_image, cv2.COLOR_GRAY2RGB)

        polygon.draw_boxes_and_save(path, rotate_image, '_first')
        polygon.group_all_blocks()

        polygon.draw_boxes_and_save(path, rotate_image, '_second')

        polygon.delete_trash_boxes()
        polygon.delete_main_borders()
        polygon.draw_boxes_and_save(path, rotate_image, '_third')

        cut_image = polygon.cut_trash(rotate_image)
        filesystem.save_image_to_disk(path, img.filename, cut_image)

    zip_file_name = filesystem.path_images_to_zip(path, images, pathname)

    response = send_from_directory(path, zip_file_name)

    return response


@app.route("/")
@app.route("/index")
def home():
    return render_template("index.html", processing=False)


if __name__ == "__main__":
    app.run(host=HTTP_HOSTNAME, port=HTTP_PORT, debug=True, use_reloader=False)
