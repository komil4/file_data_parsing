import os
import cv2
import pandas as pd
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from flask import Flask, request, json
from pdf2image import convert_from_path
import fitz


# Show image
def show_image(image, name="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(name)
    plt.show()


# Show image in full screen
def show_image_full_screen(image, name="Image"):
    cv2.imshow(name, image)
    cv2.waitKey(0)


# Save image to disk
def save_image_to_disk(image, name):
    cv2.imwrite(name, image)


# Split PDF file to few images
def split_pdf_doc(filename, method=0):
    file_names = []
    print("Start convert PDF file " + filename + " to JPEG")
    if method == 0:
        pages = []
        doc = fitz.open(filename)
        for n in range(doc.pageCount):
            page = doc.loadPage(n)
            pix = page.getPixmap()
            image = np.frombuffer(pix.samples,
                                  dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            image = np.ascontiguousarray(image[..., [2, 1, 0]])
            pages.append(image)

        for i in range(len(pages)):
            page_filename = filename.split(".")[0] + "_page_" + str(i) + ".jpeg"
            save_image_to_disk(pages[i], page_filename)
            file_names.append(page_filename)
    else:
        pages = convert_from_path(os.getcwd() + '/' + filename, size=(10000, None))
        for i in range(len(pages)):
            page_filename = filename.split(".")[0] + '_page_' + str(i) + '.jpeg'
            save_image_to_disk(pages[i], page_filename)
            file_names.append(page_filename)

    print("Convert success")

    return file_names


# Save files to disk
def save_files_to_disk(files):
    file_names = []
    for file_name in files:
        file = files.get(file_name)
        file.save(file.filename)
        file.stream.seek(0)
        file_names.append(file.filename)

    return file_names


# Get saved filenames
def get_images_from_files(file_names):
    file_names_new = []

    # Working with files
    for filename in file_names:
        # If file in PDF format
        if filename[-3:] == "pdf":
            # Split PDF to images
            page_file_names = split_pdf_doc(filename, 0)
            file_names_new.append(page_file_names)
        else:
            file_names_new.append(filename)

    return file_names_new


# Get image for pytesseract
def get_image_for_tesseract(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # show_image(kernel, "Kernel")
    # border = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
    # show_image(border, "Border")
    # resizing = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # show_image(resizing, "Resizing")
    # dilation = cv2.dilate(resizing, kernel, iterations=1)
    # show_image(dilation, "Dilation")
    # erosion = cv2.erode(dilation, kernel, iterations=2)

    erosion = cv2.erode(image, kernel)
    median = cv2.medianBlur(erosion, 3)
    dilation = cv2.dilate(median, kernel)

    # Test block
    if box_image_test:
        global box_image_iterator
        save_image_to_disk(image, image_path + str(box_image_iterator) + '_image.jpeg')
        save_image_to_disk(kernel, image_path + str(box_image_iterator) + '_kernel.jpeg')
        save_image_to_disk(median, image_path + str(box_image_iterator) + '_median.jpeg')
        save_image_to_disk(erosion, image_path + str(box_image_iterator) + '_erosion.jpeg')
        save_image_to_disk(dilation, image_path + str(box_image_iterator) + '_dilation.jpeg')
        box_image_iterator = box_image_iterator + 1

    return dilation


# Get single data from file
def get_single_data_from_image(image):
    new_image = get_image_for_tesseract(image)

    # Test block
    if box_image_test:
        global box_image_iterator
        save_image_to_disk(new_image, image_path + str(box_image_iterator) + '.jpeg')
        box_image_iterator = box_image_iterator + 1

    custom_config = '--psm 7 --oem 1'
    out = pytesseract.image_to_string(new_image, lang='rus+eng', config=custom_config)
    if len(out) == 0:
        custom_config = '--psm 8 --oem 3'
        out = pytesseract.image_to_string(new_image, lang='rus+eng', config=custom_config)
        # show_image(erosion, "Erosion")
    if len(out) == 0:
        custom_config = '--psm 8 --oem 1'
        out = pytesseract.image_to_string(new_image, lang='rus+eng', config=custom_config)
        # show_image(erosion, "Erosion")
    if len(out) == 0:
        out = ''

    return out


# Work with image
def get_data_from_files(files, datatype=0):
    files_data = {}
    file_names = get_images_from_files(files)

    # Working with files
    for filename in file_names:
        # Get main image
        image, image_bin = get_main_image(filename, 90)
        # For tests
        # show_image(img)
        # save_image_to_disk(img, '/Users/kamil/PycharmProjects/pythonProject/test.jpeg')

        print("Starting get data for file " + filename)
        try:
            if datatype == 1:
                file_data = {filename: get_table_data_from_image(image, image_bin)}
            else:
                file_data = {filename: get_single_data_from_image(image)}
            files_data.update(file_data)
        except:
            print("Failed get data for file " + filename + "!")
            continue

    return files_data


# Get main image
def get_main_image(file, rotation=90):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
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

    #save_image_to_disk(img_bin, '/Users/kamil/PycharmProjects/pythonProject/images/bin.jpeg')
    #save_image_to_disk(rotated, '/Users/kamil/PycharmProjects/pythonProject/images/img.jpeg')

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 9)
    #image = cv2.adaptiveThreshold(th, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    #dilation = cv2.dilate(th, kernel)
    #erosion = cv2.erode(dilation, kernel, iterations=3)
    #median = cv2.medianBlur(erosion, 3)

    #erosion_median = cv2.bitwise_and(erosion, median)
    #summary = cv2.bitwise_and(erosion_median, dilation)

    #g2 = cv2.medianBlur(th, 3)

    #image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 9)
    #ret2, image = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Test block
    if main_image_test:
        global main_image_iterator
        #save_image_to_disk(erosion, image_path + str(main_image_iterator) + '_erosion.jpeg')
        #save_image_to_disk(dilation, image_path + str(main_image_iterator) + '_dilation.jpeg')
        #save_image_to_disk(median, image_path + str(main_image_iterator) + '_median.jpeg')
        #save_image_to_disk(th, image_path + str(main_image_iterator) + '_th.jpeg')
        #save_image_to_disk(img, image_path + str(main_image_iterator) + '_main.jpeg')
        #save_image_to_disk(image, image_path + str(main_image_iterator) + '_threshold.jpeg')
        #save_image_to_disk(summary, image_path + str(main_image_iterator) + '_summary.jpeg')
        main_image_iterator = main_image_iterator + 1

    return rotated, img_bin


# Get table lines from file
def get_lines(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_bin = 255 - img
    #thresh, img_for_lines = cv2.threshold(img_bin, 128, 255, cv2.THRESH_BINARY)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1] // 150))
    eroded_vertical_image = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines = cv2.dilate(eroded_vertical_image, vertical_kernel, iterations=5)

    #save_image_to_disk(vertical_lines, '/Users/kamil/PycharmProjects/pythonProject/images/vert.jpeg')

    # thresh, vertical_lines = cv2.threshold(vertical_lines, 128, 255, cv2.THRESH_BINARY)

    # save_image_to_disk(vertical_lines, '/Users/kamil/PycharmProjects/pythonProject/images/vert2.jpeg')

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1] // 150, 1))
    eroded_horizontal_image = cv2.erode(img_bin, horizontal_kernel, iterations=3)
    horizontal_lines = cv2.dilate(eroded_horizontal_image, horizontal_kernel, iterations=5)

    #save_image_to_disk(horizontal_lines, '/Users/kamil/PycharmProjects/pythonProject/images/hor.jpeg')

    # thresh, horizontal_lines = cv2.threshold(horizontal_lines, 128, 255, cv2.THRESH_BINARY)

    # save_image_to_disk(horizontal_lines, '/Users/kamil/PycharmProjects/pythonProject/images/hor2.jpeg')

    vertical_horizontal_lines = cv2.bitwise_or(vertical_lines, horizontal_lines)
    # vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

    #save_image_to_disk(vertical_horizontal_lines, '/Users/kamil/PycharmProjects/pythonProject/images/ver_hor.jpeg')

    #vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=5)

    save_image_to_disk(vertical_horizontal_lines, '/Users/kamil/PycharmProjects/pythonProject/images/lines_1.jpeg')

    vertical_horizontal_lines_dilate = cv2.dilate(vertical_horizontal_lines, kernel, iterations=5)
    save_image_to_disk(vertical_horizontal_lines_dilate, '/Users/kamil/PycharmProjects/pythonProject/images/lines_2.jpeg')
    vertical_horizontal_lines_erode = cv2.erode(vertical_horizontal_lines_dilate, kernel, iterations=5)
    save_image_to_disk(vertical_horizontal_lines_erode, '/Users/kamil/PycharmProjects/pythonProject/images/lines_3.jpeg')
    thresh, vertical_horizontal_lines_tr = cv2.threshold(vertical_horizontal_lines_erode, 50, 255, cv2.THRESH_BINARY)
    vertical_horizontal_lines = cv2.dilate(vertical_horizontal_lines_tr, kernel, iterations=1)
    # vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=1)

    print("Lines created")

    return vertical_horizontal_lines


def get_lines_2(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.dilate(img, kernel, iterations=2)

    structuring_element = np.ones((1, 50), np.uint8)
    erode_image = cv2.erode(img, structuring_element, iterations=1)
    horizontal_lines = cv2.dilate(erode_image, structuring_element, iterations=1)

    structuring_element = np.ones((50, 1), np.uint8)
    erode_image = cv2.erode(img, structuring_element, iterations=1)
    vertical_lines = cv2.dilate(erode_image, structuring_element, iterations=1)

    structuring_element = np.ones((3, 3), np.uint8)
    merge_image = horizontal_lines + vertical_lines
    merge_image = cv2.dilate(merge_image, structuring_element, iterations=2)

    return merge_image


# Get image without lines
def get_image_without_lines(img, vertical_horizontal_lines):
    # thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines, 128, 255,
                                                      #cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # image_and = cv2.bitwise_or(img, vertical_horizontal_lines)
    image_xor = cv2.bitwise_xor(img, vertical_horizontal_lines)
    image_without_lines = 255 - image_xor

    return image_without_lines


def get_boxes(vertical_horizontal_lines):
    vertical_horizontal_lines = 255 - vertical_horizontal_lines
    save_image_to_disk(vertical_horizontal_lines, '/Users/kamil/PycharmProjects/pythonProject/images/lines_end.jpeg')
    contours, hierarchy = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes), key=lambda box: box[1][1]))

    bounding_boxes_new = []
    for box in bounding_boxes:
        if box[2] > 50 and box[3] > 50:
            bounding_boxes_new.append(box)

    # Find start table
    y_list = [bounding_boxes_new[i][1] for i in range(len(bounding_boxes_new))]
    y_include_list = []
    i = 0
    while i < len(y_list) / 2:
        count = 1
        for j in range(i + 1, len(y_list)):
            if abs(y_list[j - 1] - y_list[j]) <= 10:
                count = count + 1
            else:
                break
        for q in range(1, count + 1):
            y_include_list.append(count)
        i = i + count
    max_count_y = max(y_include_list)
    min_y_table = 0
    for i in range(len(y_include_list)):
        if y_include_list[i] == max_count_y and min_y_table == 0:
            min_y_table = i
    start_y_position = y_list[min_y_table]

    # Find end table
    y_list_reverse = list(reversed(y_list))
    y_include_list_reverse = list(reversed(y_include_list))
    max_y_table = 0
    for i in range(len(y_include_list_reverse)):
        if y_include_list_reverse[i] == max_count_y and max_y_table == 0:
            max_y_table = i
    end_y_position = y_list_reverse[max_y_table]

    boxes = []
    weights = []
    heights = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # boxes.append([x, y, w, h])
        if 5000 > w > 70 and h > 20 and start_y_position <= y <= end_y_position:
            weights.append(w)
            heights.append(h)
            boxes.append([x, y, w, h])
            # for show boxes
            # image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
            # show_image(image)

    # show_image_full_screen(image)

    return boxes, bounding_boxes_new


def get_boxes_list(vertical_horizontal_lines):
    boxes, bounding_boxes = get_boxes(vertical_horizontal_lines)

    rows = []
    columns = [boxes[0]]

    sum_y = boxes[0][1]
    left = 0
    right = 0
    first_row = []
    count = 1
    for i in range(1, len(boxes)):
        if boxes[i][0] <= left != 0 or boxes[i][0] + boxes[i][2] > right != 0:
            continue
        if abs(sum_y / count - boxes[i][1]) > 20:
            if len(first_row) == 0:
                rows.append(columns)
                first_row = columns
            elif len(first_row) == len(columns):
                rows.append(columns)
            columns = [boxes[i]]
            count = 1
            sum_y = boxes[i][1]
        else:
            columns.append(boxes[i])
            sum_y = sum_y + boxes[i][1]
            count = count + 1
        if i + 1 == len(boxes):
            if len(first_row) == 0:
                rows.append(columns)
            elif len(first_row) == len(columns):
                rows.append(columns)
            break
        if len(rows) == 1:
            left_list = []
            right_list = []
            for c in rows[0]:
                left_list.append(c[0])
                right_list.append(c[0] + c[2])
            left = min(left_list) - 50
            right = max(right_list) + 50

    boxes_list = []
    for i in range(len(rows)):
        boxes_new = []
        x_list = [rows[i][j][0] for j in range(len(rows[i]))]
        x_list.sort()
        for x in x_list:
            for cell in rows[i]:
                if cell[0] == x:
                    boxes_new.append(cell)
        boxes_list.append(boxes_new)

    # for box in boxes_list:
    # print(box)

    return boxes_list


def get_data_frame(image_without_lines, boxes_list):
    dataframe_final = []
    columns = []
    for i in range(len(boxes_list)):
        q = 0
        strings = []
        for j in range(len(boxes_list[i])):
            if len(boxes_list[i][j]) == 0:
                dataframe_final.append(' ')
            else:
                y, x, w, h = boxes_list[i][j][0], boxes_list[i][j][1], boxes_list[i][j][2], \
                             boxes_list[i][j][3]
                roi = image_without_lines[x:x + h, y:y + w]
                out = get_single_data_from_image(roi)
                strings.append(out)
                if i == 0:
                    columns.append(str(q))
                    q = q + 1
                # print(out)
                # if(len(out)==0):
                # out = pytesseract.image_to_string(erosion, lang='rus+eng')
                # s = s + " " + out
                # print(s)
        dataframe_final.append(strings)
    # print(dataframe_final)

    dataframe = pd.DataFrame(np.array(dataframe_final), columns=columns)
    dictionary = dataframe.to_dict()
    # print(dataframe)
    return dictionary


def find_cell_contours(image, lines):
    black_pixels = np.where(lines == 255)
    y = black_pixels[0]
    x = black_pixels[1]
    for i in range(len(y)):
        image[y[i]][x[i]] = 255

    return image


# Get table data from file
def get_table_data_from_image(image, image_bin):
    save_image_to_disk(image, '/Users/kamil/PycharmProjects/pythonProject/images/image.jpeg')
    # Get lines from file
    vertical_horizontal_lines = get_lines(image)
    # vertical_horizontal_lines = get_lines_2(image_bin)
    # For tests
    # show_image(vertical_horizontal_lines)
    save_image_to_disk(vertical_horizontal_lines, '/Users/kamil/PycharmProjects/pythonProject/images/lines.jpeg')
    save_image_to_disk(image, '/Users/kamil/PycharmProjects/pythonProject/images/img.jpeg')

    # Get image without lines
    image_without_lines = find_cell_contours(image, vertical_horizontal_lines)
    # image_without_lines = get_image_without_lines(image, vertical_horizontal_lines)
    # For tests
    # show_image(image_without_lines)
    save_image_to_disk(image_without_lines, '/Users/kamil/PycharmProjects/pythonProject/images/without_lines.jpeg')

    # Main logic
    print("Start get dataframe")
    boxes_list = get_boxes_list(vertical_horizontal_lines)
    dataframe = get_data_frame(image, boxes_list)
    print("Dataframe getting successful")

    return dataframe


# HTTP service methods
app = Flask(__name__)


@app.route("/getTableDataFromFiles", methods=["POST"])
def get_table_data_from_files():
    file_names = save_files_to_disk(request.files)
    files_data = get_data_from_files(file_names, 1)
    print("Send response to client")
    json_string = json.dumps(files_data)
    return json_string


@app.route("/getSingleDataFromFiles", methods=["POST"])
def get_single_data_from_files():
    file_names = save_files_to_disk(request.files)
    files_data = get_data_from_files(file_names, 0)
    print("Send response to client")
    json_string = json.dumps(files_data)
    return json_string


@app.route("/")
def home():
    return "Hello, it's OCB program!"


# Image test block
box_image_test = False
box_image_iterator = 0
image_path = '/Users/kamil/PycharmProjects/pythonProject/images/'

main_image_test = True
main_image_iterator = 0

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=3000, debug=True, use_reloader=False)