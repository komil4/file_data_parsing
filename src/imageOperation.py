import cv2
import numpy as np
import filesystem

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
def convert_to_grayscale_images(images, save=True):
    for image in images:
        image["image"] = cv2.cvtColor(image.get("image"), cv2.COLOR_BGR2GRAY)

        if save:
            filesystem.save_image_to_disk(image.get("path"), image.get("name"), image.get("image"))


# Get rotate image
def threshold_images(images, save=True):
    for image in images:
        img, img_bin = cv2.threshold(image.get("image"), 128, 255, cv2.THRESH_BINARY)
        image["image"] = img

        if save:
            filesystem.save_image_to_disk(image.get("path"), image.get("name"), image.get("image"))
