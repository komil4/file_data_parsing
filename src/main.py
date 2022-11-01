import os
import argparse
import cv2

import imageOperation
import filesystem

from flask import Flask, request, json, render_template, url_for, send_from_directory

from matplotlib import pyplot as plt


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
parser.add_argument("-port", default='3000', help='This is a port value. Default set 3000')
args = parser.parse_args()
# Global values
hostname = args.host
port = args.port

temp_path = (os.getcwd() + '/temp\\').replace('\\', '/')


# Legacy block
# Global values
files_data_for_interface = {}
file_names_for_interface = []
result_file_names_for_interface = []
data = []


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

    pathname = filesystem.get_new_path_name(temp_path)
    path = temp_path + pathname + "/"

    file_names = filesystem.save_files_to_disk(path, request.files)
    images = filesystem.get_images_from_files(path, file_names)
    if color:
        imageOperation.convert_to_grayscale_images(images, False)
    if threshold:
        imageOperation.threshold_images(images)

    zip_file_name = filesystem.path_images_to_zip(path, images)

    response = send_from_directory(path, zip_file_name)
    # delete_file(zip_file_name)

    return response


@app.route("/getRotateImages/<degree>", methods=["POST"])
def get_rotate_images(degree):
    pathname = filesystem.get_new_path_name(temp_path)
    path = temp_path + pathname + "/"

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


# -------------------------------------
# Блок тестирования
# Image test block
# Чекбокс для тестирования блоков
box_image_test = False
box_image_iterator = 0


# Чекбокс для тестирования изначальных изображений
main_image_test = True
main_image_iterator = 0

if __name__ == "__main__":
    app.run(host=hostname, port=port, debug=True, use_reloader=False)
