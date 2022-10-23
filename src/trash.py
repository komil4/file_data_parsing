img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th3 = cv2.adaptiveThreshold(img_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

##Otsu
ret1,th1 = cv2.threshold(img_grey,127,255,cv2.THRESH_BINARY)

ret2,th2 = cv2.threshold(img_grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(img_grey,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

##Last

#img = img_for_lines

total_cells = 0
    center = 0
    for row in rows:
        total_cells = 0
        for i in range(len(row)):
            if len(row[i]) > total_cells:
                total_cells = len(row[i])

                center = [int(rows[i][j][0] + rows[i][j][2] / 2) for j in range(len(rows[i])) if rows[0]]
                center = np.array(center)
                center.sort()

    boxes_list = []
    for i in range(len(rows)):
        l = []
        for k in range(total_cells):
            l.append([])
        for j in range(len(rows[i])):
            diff = abs(center - (rows[i][j][0] + rows[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            l[j][indexing].append(rows[i][j])
        boxes_list.append(l)

previous = boxes[0]
for i in range(1, len(boxes)):
    if (boxes[i][1] <= previous[1] + meanHeight / 2):
        columns.append(boxes[i])
        previous = boxes[i]
        if (i == len(boxes) - 1):
            rows.append(columns)
    else:
        rows.append(columns)
        columns = []
        previous = boxes[i]
        columns.append(boxes[i])