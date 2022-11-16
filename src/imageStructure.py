import cv2
import filesystem


class Polygon:
    def __init__(self, image, name, uid='0'):
        self.uid = uid
        self.name = name
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.contour_image = image
        image = 255 - image
        rt, th = cv2.threshold(image, 127, 255, 0)
        contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes), key=lambda box: box[1][1]))

        self.boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box = [x, y, w, h]
            self.boxes.append(box)

        self.contours = contours
        self.bounding_boxes = bounding_boxes
        self.dif = 40
        self.group_boxes = self.boxes.copy()

    def draw_contours_and_save(self, path, image):
        image_copy = image.copy()
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            # for show boxes
            image_copy = cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 5)

        filesystem.save_image_to_disk(path, self.uid + "_" + self.name + ".jpg", image_copy)

    def draw_boxes_and_save(self, path, image, part=''):
        image_copy = image.copy()
        for box in self.group_boxes:
            # for show boxes
            image_copy = cv2.rectangle(image_copy, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 5)

        filesystem.save_image_to_disk(path, self.uid + "_" + self.name + part + ".jpg", image_copy)

    def group_all_blocks(self):
        for i in range(len(self.group_boxes)):
            while self.add_top_block(i):
                continue
        for i in range(len(self.group_boxes)):
            while self.find_right_block(i):
                continue

        new_group_boxes = []
        for box in self.group_boxes:
            if box[0] != 0 and box[1] != 0 and box[2] != 0 and box[3] != 0:
                new_group_boxes.append(box)

        self.group_boxes = new_group_boxes

    def add_top_block(self, box_i):
        box = self.group_boxes[box_i]
        for i in range(len(self.group_boxes)):
            if self.group_boxes[i][0] == 0 and self.group_boxes[i][1] == 0 and self.group_boxes[i][2] == 0 and \
                    self.group_boxes[i][3] == 0:
                continue
            if abs(self.group_boxes[i][0] - box[0]) < self.dif \
                    and abs((self.group_boxes[i][1] + self.group_boxes[i][3]) - box[1]) < self.dif \
                    and abs((self.group_boxes[i][0] + self.group_boxes[i][2]) - (box[0] + box[2])) < self.dif \
                    and i != box_i:

                # Расчет высоты блока
                self.group_boxes[box_i][3] = self.group_boxes[box_i][1] - self.group_boxes[i][1] + \
                                             self.group_boxes[box_i][3]

                # Установка x и y
                if self.group_boxes[box_i][0] > self.group_boxes[i][0]:
                    self.group_boxes[box_i][0] = self.group_boxes[i][0]
                if self.group_boxes[box_i][1] > self.group_boxes[i][1]:
                    self.group_boxes[box_i][1] = self.group_boxes[i][1]

                # Обнуление ячейки
                self.group_boxes[i][0] = 0
                self.group_boxes[i][1] = 0
                self.group_boxes[i][2] = 0
                self.group_boxes[i][3] = 0

                return True

        return False

    def find_right_block(self, box_i):
        box = self.group_boxes[box_i]
        for i in range(len(self.group_boxes)):
            if self.group_boxes[i][0] == 0 and self.group_boxes[i][1] == 0 and self.group_boxes[i][2] == 0 and \
                    self.group_boxes[i][3] == 0:
                continue
            if abs(box[1] - self.group_boxes[i][1]) < self.dif \
                    and abs((box[0] + box[2]) - self.group_boxes[i][0]) < self.dif \
                    and abs((box[1] + box[3]) - (self.group_boxes[i][1] + self.group_boxes[i][3])) < self.dif \
                    and i != box_i:

                # Расчет ширины блока
                self.group_boxes[box_i][2] = self.group_boxes[i][0] - self.group_boxes[box_i][0] + self.group_boxes[i][2]

                max_y = self.group_boxes[box_i][1] + self.group_boxes[box_i][3]\
                if self.group_boxes[box_i][1] + self.group_boxes[box_i][3] > self.group_boxes[i][1] + self.group_boxes[i][3]\
                    else self.group_boxes[i][1] + self.group_boxes[i][3]

                # Установка x и y
                if self.group_boxes[i][0] < self.group_boxes[box_i][0]:
                    self.group_boxes[box_i][0] = self.group_boxes[i][0]
                if self.group_boxes[i][1] < self.group_boxes[box_i][1]:
                    self.group_boxes[box_i][1] = self.group_boxes[i][1]

                # Установка высоты
                self.group_boxes[box_i][3] = max_y - self.group_boxes[box_i][1]

                # Обнуление ячейки
                self.group_boxes[i][0] = 0
                self.group_boxes[i][1] = 0
                self.group_boxes[i][2] = 0
                self.group_boxes[i][3] = 0

                return True

        return False

    def cut_trash(self, image):
        boxes = self.group_boxes

        dif = 20

        block = [0, 0, 0, 0, 0]
        for i in range(len(boxes)):
            if boxes[i][2] * boxes[i][3] > block[4]:
                block[0] = 0 if boxes[i][0] - dif < 0 else boxes[i][0] - dif
                block[1] = 0 if boxes[i][1] - dif < 0 else boxes[i][1] - dif
                block[2] = image.shape[1] - boxes[i][0] if boxes[i][0] + boxes[i][2] + 2 * dif > image.shape[1] else boxes[i][2] + 2 * dif
                block[3] = image.shape[0] - boxes[i][1] if boxes[i][1] + boxes[i][3] + 2 * dif > image.shape[0] else boxes[i][3] + 2 * dif
                block[4] = block[2] * block[3]

        crop_image = image[block[1]:block[1] + block[3], block[0]:block[0] + block[2]]
        return crop_image

    def delete_main_borders(self):
        boxes = self.group_boxes
        new_boxes = []
        for i in range(len(boxes)):
            add = True
            for j in range(len(boxes)):
                if i == j:
                    continue
                if (boxes[i][0] < boxes[j][0] + self.dif) and (boxes[i][1] < boxes[j][1] + self.dif) \
                        and (boxes[i][0] + boxes[i][2] > boxes[j][0] + boxes[j][2] - self.dif) \
                        and (boxes[i][1] + boxes[i][3] > boxes[j][1] + boxes[j][3] - self.dif):
                    add = False
                    break

            if add:
                new_boxes.append(boxes[i])

        self.group_boxes = new_boxes

    def delete_trash_boxes(self):
        boxes = self.group_boxes
        new_boxes = []
        for i in range(len(boxes)):
            if boxes[i][2] > self.dif and boxes[i][3] > self.dif:
                new_boxes.append(boxes[i])

        self.group_boxes = new_boxes

