import cv2
import numpy as np
import filesystem
from statistics import mean

# -------------------------------------
# Обработка изображениq
# Get main image
def get_main_image(image, rotation=90):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Image rotation
    if rotation == 90:
        print("Rotate image")
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    ret, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    coords = np.column_stack(np.where(img_bin == 255))
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle = angle - 90
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(img, rotation_matrix, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    img_bin = cv2.warpAffine(img_bin, rotation_matrix, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    ret, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # save_image_to_disk(img_bin, '/Users/kamil/PycharmProjects/pythonProject/images/bin.jpeg')
    # save_image_to_disk(rotated, '/Users/kamil/PycharmProjects/pythonProject/images/img.jpeg')

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 9)
    # image = cv2.adaptiveThreshold(th, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    # dilation = cv2.dilate(th, kernel)
    # erosion = cv2.erode(dilation, kernel, iterations=3)
    # median = cv2.medianBlur(erosion, 3)

    # erosion_median = cv2.bitwise_and(erosion, median)
    # summary = cv2.bitwise_and(erosion_median, dilation)

    # g2 = cv2.medianBlur(th, 3)

    # image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 9)
    # ret2, image = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Test block
    #if main_image_test:
        #global main_image_iterator
        # save_image_to_disk(erosion, image_path + str(main_image_iterator) + '_erosion.jpeg')
        # save_image_to_disk(dilation, image_path + str(main_image_iterator) + '_dilation.jpeg')
        # save_image_to_disk(median, image_path + str(main_image_iterator) + '_median.jpeg')
        # save_image_to_disk(th, image_path + str(main_image_iterator) + '_th.jpeg')
        # save_image_to_disk(img, image_path + str(main_image_iterator) + '_main.jpeg')
        # save_image_to_disk(image, image_path + str(main_image_iterator) + '_threshold.jpeg')
        # save_image_to_disk(summary, image_path + str(main_image_iterator) + '_summary.jpeg')
        #main_image_iterator = main_image_iterator + 1

    return rotated, img_bin


# Get rotate image
def get_rotate_image(image, rotation='0'):
    if rotation == '90':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == '180':
        return cv2.rotate(image, cv2.ROTATE_180)
    elif rotation == '270':
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image


# Get rotate image
def rotate_images(images, rotation='0', save=True):
    for image in images:
        if rotation == '90':
            image["image"] = cv2.rotate(image.get("image"), cv2.ROTATE_90_CLOCKWISE)
        elif rotation == '180':
            image["image"] = cv2.rotate(image.get("image"), cv2.ROTATE_180)
        elif rotation == '270':
            image["image"] = cv2.rotate(image.get("image"), cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return
        if save:
            filesystem.save_image_to_disk(image.get("path"), image.get("name"), image.get("image"))


# Get rotate image
def auto_rotate_images(images, save=True):
    for image in images:
        img = image.get("image")
        if img.shape[0] > img.shape[1]:
            image["image"] = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if save:
                filesystem.save_image_to_disk(image.get("path"), image.get("name"), image.get("image"))


# Get rotate image
def convert_to_grayscale_images(images, save=True):
    for image in images:
        image["image"] = cv2.cvtColor(image.get("image"), cv2.COLOR_BGR2GRAY)

        if save:
            filesystem.save_image_to_disk(image.get("path"), image.get("name"), image.get("image"))


# Get rotate image
def threshold_images(images, save=True):
    for image in images:
        gray = cv2.cvtColor(image.get("image"), cv2.COLOR_BGR2GRAY)

        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 299, 15)

        img = cv2.GaussianBlur(th, (3, 3), 0)
        
        #ret3, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)

        #ret, img = cv2.threshold(th3, 160, 255, cv2.THRESH_TOZERO)
        '''
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(50, 50))
        lab = cv2.cvtColor(image.get("image"), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        #ret, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)

        #kernel = np.ones((2, 2), np.uint8)
        #obr_img = cv2.erode(thresh, kernel, iterations=1)

        img = cv2.GaussianBlur(gray, (3, 3), 0)

        # img = 255 - img;

        # img, img_bin = cv2.threshold(image.get("image"), 128, 255, cv2.THRESH_BINARY)
        '''



        image["image"] = img

        if save:
            filesystem.save_image_to_disk(image.get("path"), image.get("name"), image.get("image"))


def get_pixel_size(image):
    if len(image.shape) > 2:
        return 1;
    width = image.shape[0]
    height = image.shape[1]
    pixel_sizes = []
    count = 0
    for x in range(1, width):
        for y in range(1, height):
            if image[x, y] != image[x - 1, y] and count > 1 and abs(int(image[x, y]) - int(image[x - 1, y])) > 5:
                pixel_sizes.append(count)
                count = 0
            else:
                count += 1
        count = 0;

    return mean(pixel_sizes)


def get_vertical_lines(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.dilate(image, kernel, iterations=2)
    structuring_element = np.ones((200, 2), np.uint8)
    erode_image = cv2.erode(img, structuring_element, iterations=1)
    vertical_lines = cv2.dilate(erode_image, structuring_element, iterations=1)

    contours, hierarchy = cv2.findContours(vertical_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filesystem.save_image_to_disk(image.get("path"), image.get("name"), "lines.jpeg")

    return vertical_lines

def delete_trash_tables_from_images(images, save=True):
    for image in images:
        img = image.get("image")
        img_inv = 255 - img
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_a = cv2.dilate(img_inv, kernel, iterations=3)
        structuring_element = np.ones((100, 1), np.uint8)
        erode_image = cv2.erode(img_a, structuring_element, iterations=1)
        vertical_lines = cv2.dilate(erode_image, structuring_element, iterations=5)

        # contours, hierarchy = cv2.findContours(vertical_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        lines = cv2.HoughLinesP(vertical_lines, 1, np.pi / 360, threshold=120,  # Min number of votes for valid line
                                minLineLength=vertical_lines.shape[0] // 2,  # Min allowed length of line
                                maxLineGap=20)

        #color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        y_min_max = [vertical_lines.shape[0], 0]
        x_min = vertical_lines.shape[1]

        top_point = []
        down_point = []

        lines_list = []
        for points in lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            '''
            if y1 < y_min_max[0] and x_min > x1:
                top_point = [x1, y1]
                y_min_max[0] = y1
                x_min = x1 + 100
            if y2 < y_min_max[0] and x_min > x2:
                top_point = [x2, y2]
                y_min_max[0] = y2
                x_min = x2 + 100
    
            if y1 > y_min_max[1] and x_min > x1:
                down_point = [x1, y1]
                y_min_max[1] = y1
                x_min = x1 + 100
            if y2 > y_min_max[1] and x_min > x1:
                down_point = [x2, y2]
                y_min_max[1] = y2
                x_min = x2 + 100
            '''
            # Draw the lines joing the points
            # On the original image
            # if abs(y1 - y2) > vertical_lines.shape[0] // 2:
            # cv2.line(color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Maintain a simples lookup list for points
            lines_list.append([(x1, y1), (x2, y2)])
            if x1 < x_min:
                x_min = x1
            if x2 < x_min:
                x_min = x2

        points_for_draw = np.array([[0, 0],
                                    [x_min, 0],
                                    [x_min, vertical_lines.shape[0]],
                                    [0, vertical_lines.shape[0]]])
        points_for_draw.reshape(-1, 1, 2)

        cv2.fillPoly(img, [points_for_draw], color=(255, 255, 255))

        # filesystem.save_image_to_disk(image.get("path"), "img.jpeg", color)

        # filesystem.save_image_to_disk(image.get("path"), "vertical_lines.jpeg", vertical_lines)

        image["image"] = img

        if save:
            filesystem.save_image_to_disk(image.get("path"), image.get("name"), image.get("image"))

    return
