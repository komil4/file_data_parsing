import cv2
import numpy as np
from pathlib import PurePosixPath
from PIL import Image
from statistics import mean
import logging

import filesystem

IMAGE_WIDTH = 5000


def images_process(images, steps):
    for img in images:
        for step in steps:
            img.get_functions()[step.get('name')]\
                (step.get('parameters')[0] if len(step.get('parameters', [])) > 0 and step.get('parameters')[
                    0] != '' else '',
                 step.get('parameters')[1] if len(step.get('parameters', [])) > 1 and step.get('parameters')[
                     1] != '' else '')


def get_lines(image, lines_cort=(200, 200), dilate_cort=(5, 5), erode_cort=(5, 5)):
    image = 255 - image

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.dilate(image, kernel, iterations=1)
    structuring_element = np.ones(lines_cort, np.uint8)
    erode_image = cv2.erode(img, structuring_element, iterations=1)
    lines = cv2.dilate(erode_image, structuring_element, iterations=1)

    kernel = np.ones(dilate_cort, np.uint8)
    lines = cv2.dilate(lines, kernel, iterations=2)

    kernel = np.ones(erode_cort, np.uint8)
    lines = cv2.erode(lines, kernel, iterations=1)

    return lines


def get_new_image(height, width):
    return np.array(Image.new('RGB', (width, height), (255, 255, 255)))


class Img:
    def __init__(self, path, filename, uid='0'):
        stream = open(path + filename, 'rb')
        bytes = bytearray(stream.read())
        array = np.asarray(bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        self.uid = uid
        self.path = path
        self.filename = filename
        self.main_image = image
        self.name = PurePosixPath(filename).stem
        self.suffix = PurePosixPath(filename).suffix
        self.images = [{'step': 'main', 'image': image}]
        self.main_channels = len(image.shape)

    def save_image(self, from_image='', name_from_step=False):
        image = self.get_image_by_step_name(from_image)
        #
        # image = self.images.get(from_image, self.main_image)
        if name_from_step or name_from_step == 'True':
            filename = self.name + "_" + from_image + self.suffix
        else:
            filename = self.name + self.suffix
        filesystem.save_image_to_disk(self.path, filename, image)

    def gray_scale(self, from_image='', step_name='gray_scale'):
        image = self.get_image_by_step_name(from_image)

        if len(image.shape) == 2:
            logging.warning(
                'id ' + self.uid + ' - image has 2 channels, it is a grayscale. Grayscale stopped! ' + step_name)
        else:
            gray_dict = {'step': step_name if step_name != '' else 'gray_scale',
                         'image': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)}

            # Logging
            logging.info('id ' + self.uid + ' - create grayscale image ' + gray_dict.get('step'))

            self.add_image_to_step_line(gray_dict)

    def gaussian_threshold(self, from_image='main', step_name='gaussian_threshold'):
        image = self.get_image_by_step_name(from_image)
        th = cv2.adaptiveThreshold(image,
                                   255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   image.shape[1] // 25 if (image.shape[1] // 25) % 2 != 0 \
                                       else image.shape[1] // 25 + 1,
                                   15)

        th_dict = {'step': step_name if step_name != '' else 'gaussian_threshold',
                   'image': cv2.GaussianBlur(th, (3, 3), 0)}

        # Logging
        logging.info('id ' + self.uid + ' - create gaussian threshold image ' + th_dict.get('step'))

        self.add_image_to_step_line(th_dict)

    def autorotate_image(self, from_image='', step_name='autorotate_image'):
        image = self.get_image_by_step_name(from_image)
        if image.shape[0] > image.shape[1]:
            autorotate_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            autorotate_image = image

        if autorotate_image.shape[1] != IMAGE_WIDTH:
            high = IMAGE_WIDTH * autorotate_image.shape[0] // autorotate_image.shape[1]
            autorotate_image = cv2.resize(autorotate_image, (IMAGE_WIDTH, high), cv2.INTER_NEAREST)

        autorotate_dict = {'step': step_name if step_name != '' else 'autorotate_image', 'image': autorotate_image}

        # Logging
        logging.info('id ' + self.uid + ' - create autorotate image ' + autorotate_dict.get('step'))

        self.add_image_to_step_line(autorotate_dict)

    def rotate_image(self, params=[], step_name='rotated_image'):
        if len(params) > 1:
            image = self.get_image_by_step_name(params[0])
            rotation = self.get_image_by_step_name(params[1])
        else:
            return

        if rotation == '90':
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == '180':
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        elif rotation == '270':
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = image

        rotated_dict = {'step': step_name if step_name != '' else 'rotated_image',
                        'image': rotated}

        # Logging
        logging.info('id ' + self.uid + ' - rotate image ' + rotated_dict.get('step'))

        self.add_image_to_step_line(rotated_dict)

    def vertical_lines(self, from_image='', step_name='vertical_lines_negative'):
        if from_image == 'main' or from_image == '':
            self.autorotate_image('main', 'main_lines')
            self.gray_scale('main_lines', 'gray_vertical_lines')
            self.gaussian_threshold('gray_vertical_lines', 'threshold_vertical_lines')
            from_image = 'threshold_vertical_lines'
        image = self.get_image_by_step_name(from_image)

        vertical_lines = get_lines(image, lines_cort=(200, 2), dilate_cort=(2, 5), erode_cort=(10, 2))

        vertical_lines_dict = {'step': step_name if step_name != '' else 'vertical_lines_negative',
                               'image': vertical_lines}

        # Logging
        logging.info('id ' + self.uid + ' - create vertical lines image ' + vertical_lines_dict.get('step'))

        self.add_image_to_step_line(vertical_lines_dict)

    def horizontal_lines(self, from_image='', step_name='horizontal_lines_negative'):
        if from_image == 'main' or from_image == '':
            self.autorotate_image('main', 'main_lines')
            self.gray_scale('main_lines', 'gray_horizontal_lines')
            self.gaussian_threshold('gray_horizontal_lines', 'threshold_horizontal_lines')
            from_image = 'threshold_horizontal_lines'

        image = self.get_image_by_step_name(from_image)

        horizontal_lines = get_lines(image, lines_cort=(2, 200), dilate_cort=(5, 2), erode_cort=(2, 10))

        horizontal_lines_dict = {'step': step_name if step_name != '' else 'horizontal_lines_negative',
                                 'image': horizontal_lines}

        # Logging
        logging.info('id ' + self.uid + ' - create horizontal lines image ' + horizontal_lines_dict.get('step'))

        self.add_image_to_step_line(horizontal_lines_dict)

    def table_tines(self, image_names=[], step_name='table_lines'):
        if len(image_names) > 0:
            image = self.get_image_by_step_name(image_names[0])
            for i in range(1, len(image_names)):
                image = cv2.bitwise_or(image, self.get_image_by_step_name(image_names[i]))
        else:
            image = self.get_image_by_step_name()

        table_lines_dict = {'step': step_name if step_name != '' else 'table_lines',
                            'image': image}

        # Logging
        logging.info('id ' + self.uid + ' - create table lines image ' + table_lines_dict.get('step'))

        self.add_image_to_step_line(table_lines_dict)

    def align_table(self, image_names=[], step_name='align_table'):
        if len(image_names) > 1:
            lines_image = self.get_image_by_step_name(image_names[0])
            image = self.get_image_by_step_name(image_names[1])
        elif len(image_names) > 0:
            lines_image = self.get_image_by_step_name(image_names[0])
            image = self.get_image_by_step_name('autorotate_image')
        else:
            image = self.get_image_by_step_name()
            lines_image = image

        lines = cv2.HoughLinesP(lines_image, 1, np.pi / 360, threshold=120,  # Min number of votes for valid line
                                minLineLength=lines_image.shape[0] // 2,  # Min allowed length of line
                                maxLineGap=20)
        height, width = lines_image.shape[:2]

        angles = []
        for points in lines:
            x1, y1, x2, y2 = points[0]
            if (abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2) ** 0.5 < height // 3:
                continue
            if (x1 < x2 and y1 < y2) or (x1 > x2 and y1 > y2):
                angles.append(-1 * (90 - np.degrees(np.arctan(abs((y2 - y1) / (x2 - x1))))))
            else:
                angles.append(90 - np.degrees(np.arctan(abs((y2 - y1) / (x2 - x1)))))

        angle = mean(angles) if len(angles) > 1 else 0

        center_x, center_y = (width / 2, height / 2)

        m = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
        rotated = cv2.warpAffine(image, m, (width, height))

        rotated_dict = {'step': step_name if step_name != '' else 'align_table',
                        'image': rotated
                        }

        # Logging
        logging.info('id ' + self.uid + ' - align table image ' + rotated_dict.get('step'))

        self.add_image_to_step_line(rotated_dict)

    def get_functions(self):
        return {'save_image': self.save_image,
                'gray_scale': self.gray_scale,
                'gaussian_threshold': self.gaussian_threshold,
                'autorotate_image': self.autorotate_image,
                'vertical_lines': self.vertical_lines,
                'horizontal_lines': self.horizontal_lines,
                'table_tines': self.table_tines,
                'align_table': self.align_table,
                'rotate_image': self.rotate_image}

    def get_image_by_step_name(self, step_name=''):
        if step_name != '':
            for step in self.images:
                if step.get("step") == step_name:
                    return step.get("image")
        if len(self.images):
            return self.images[-1].get('image', self.main_image)
        else:
            return self.main_image

    def add_image_to_step_line(self, step_struct):
        for i in range(len(self.images)):
            if step_struct.get('step') == self.images[i].get('step'):
                self.images[i] = step_struct

        self.images.append(step_struct)

