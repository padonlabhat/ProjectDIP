import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.measure import label, regionprops
import array as arr


def peepsctive(thresh):
    corners = cv2.goodFeaturesToTrack(thresh, 4, 0.01, 100)
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
    edged = cv2.Canny(thresh, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # cv2.imwrite('output/4pre_edged.jpg', edged)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        epsilon = 0.06 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        print(approx)
    # cv2.imwrite('output/5contour.jpg', img)

    for c in cnts:  # 1
        # find bounding box coordinates
        x, y, w, h = cv2.boundingRect(c)
        print(x)
        print(y)
        print(w)
        print(h)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # find minimum area
    rect = cv2.minAreaRect(c)
    # calculate 4 coordinates of the minimum area rectangle
    box = cv2.boxPoints(rect)
    print(box)
    # casting to integers
    box = np.int64(box)  # 6
    # draw contours
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    myPoints = np.array(approx, dtype=np.int32)
    print('********************************')
    print(myPoints)

    w = 2480;
    h = 3508;
    pts1 = np.float32([myPoints[1], myPoints[0], myPoints[2], myPoints[3]])
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (w, h))
    cv2.imwrite('output/6_perspective.jpg', result)
    return result


def preimg(imS):
    cv2.imwrite('output/1img.jpg', imS)
    gray = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/2pre_blurred.jpg', gray)
    thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('output/3pre_thresh.jpg', thresh)
    return thresh


def peepsctive2(thresh2):
    corners = cv2.goodFeaturesToTrack(thresh2, 4, 0.01, 100)
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
    edged = cv2.Canny(thresh2, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cv2.imwrite('output2/edged.jpg', edged)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        epsilon = 0.06 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        print(approx)
    cv2.imwrite('output2/contour.jpg', img2)

    for c in cnts:  # 1
        # find bounding box coordinates
        x, y, w, h = cv2.boundingRect(c)
        print(x)
        print(y)
        print(w)
        print(h)
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # find minimum area
    rect = cv2.minAreaRect(c)
    # calculate 4 coordinates of the minimum area rectangle
    box = cv2.boxPoints(rect)
    print(box)
    # casting to integers
    box = np.int64(box)  # 6
    # draw contours
    cv2.drawContours(img2, [box], 0, (0, 0, 255), 2)
    myPoints = np.array(approx, dtype=np.int32)
    print('********************************')
    print(myPoints)

    w = 2480;
    h = 3508;
    pts1 = np.float32([myPoints[1], myPoints[0], myPoints[2], myPoints[3]])
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img2, matrix, (w, h))
    cv2.imwrite('output2/perspective.jpg', result)
    return result


def preimg2(imS):
    cv2.imwrite('output2/img.jpg', imS)
    gray = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output2/gray.jpg', gray)
    thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('output2/thresh.jpg', thresh)
    return thresh

x = 146
y = 250
h = 18
w = 32
ansCorrect = []
ansUser = []
sum = 0
choose = ''
choose2 = ''
img = cv2.imread('input img/test (6).jpg')
peepsctive(preimg(img))
A4 = cv2.imread('output/6_perspective.jpg')
answersheet = cv2.resize(A4, (620, 877))
cropped_image = img[80:280, 150:330]
gray = cv2.cvtColor(answersheet, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)[1]
plt.imshow(thresh)
plt.show()

for i in range(30):
    if i < 15:
        x = 146

        print('loop 1-', i)
    else:
        print('loop 2-', i)
        if i == 15:
            y = 250

        x = 366
    print('ข้อที่ ', i + 1)
    print('X,y: ', x, y)
    num = 0
    for j in range(4):
        cropped_image = thresh[y:y + h, x:x + w]
        L = label(cropped_image)
        props = regionprops(L)

        print('ตัวเลือกที่ ', j + 1, 'sum : ', props.__len__())
        cv2.rectangle(answersheet, (x, y), (x + w, y + h), (0, 0, 255), 1)

        if props.__len__() == 4:
            cv2.rectangle(answersheet, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if j + 1 == 1:
                choose = 'A'
            if j + 1 == 2:
                choose = 'B'
            if j + 1 == 3:
                choose = 'C'
            if j + 1 == 4:
                choose = 'D'
            num += 1
        elif props.__len__() == 6 and num == 1:
            cv2.rectangle(answersheet, (x, y), (x + w, y + h), (255, 0, 0), 1)
            num = 1
        elif props.__len__() == 6 and num == 0:
            cv2.rectangle(answersheet, (x, y), (x + w, y + h), (255, 0, 0), 1)
            num = 0
        elif props.__len__() != 1:
            cv2.rectangle(answersheet, (x, y), (x + w, y + h), (255, 10, 255), 1)
            num = 5

        x += 45

    if num == 1:
        ansCorrect.append(choose)
    elif 1 < num <= 4:
        ansCorrect.append('More')
    elif num == 0:
        ansCorrect.append('NF')
    else:
        ansCorrect.append('ERROR')

    y += 26

choose2 = ''
img2 = cv2.imread('input2/test_4.jpg')
cal = peepsctive2(preimg2(img2))
A4 = cv2.imread('output2/perspective.jpg')
answersheet2 = cv2.resize(A4, (620, 877))
cropped_image = img2[80:280, 150:330]
gray = cv2.cvtColor(answersheet2, cv2.COLOR_BGR2GRAY)
thresh2 = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)[1]
plt.imshow(thresh2)
plt.show()

x = 146
y = 250
for k in range(30):
    if k < 15:
        x = 146

        print('loop 1-', k)
    else:
        print('loop 2-', k)
        if k == 15:
            y = 250

        x = 366
    print('ข้อที่ ', k + 1)
    num = 0
    for l in range(4):
        cropped_image = thresh2[y:y + h, x:x + w]
        K = label(cropped_image)
        props = regionprops(K)

        print('ตัวเลือกที่ ', l + 1, 'sum : ', props.__len__())
        cv2.rectangle(answersheet2, (x, y), (x + w, y + h), (0, 0, 255), 1)

        if props.__len__() == 4:
            cv2.rectangle(answersheet2, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if l + 1 == 1:
                choose2 = 'A'
            if l + 1 == 2:
                choose2 = 'B'
            if l + 1 == 3:
                choose2 = 'C'
            if l + 1 == 4:
                choose2 = 'D'
            num += 1
        elif props.__len__() == 6 and num == 1:
            cv2.rectangle(answersheet2, (x, y), (x + w, y + h), (255, 0, 0), 1)
            num = 1
        elif props.__len__() == 6 and num == 0:
            cv2.rectangle(answersheet2, (x, y), (x + w, y + h), (255, 0, 0), 1)
            num = 0
        elif props.__len__() != 1:
            cv2.rectangle(answersheet2, (x, y), (x + w, y + h), (255, 10, 255), 1)
            num = 5

        x += 45

    if num == 1:
        ansUser.append(choose2)
    elif 1 < num <= 4:
        ansUser.append('More')
    elif num == 0:
        ansUser.append('NF')
    else:
        ansUser.append('ERROR')

    y += 26

print('ansCorrect ', ansCorrect)
print('ansUser', ansUser)
for i in range(30):
    if ansCorrect[i]==ansUser[i]:
        sum+=1
    if i >= 3:
        if ansCorrect[i] == ansCorrect[i - 1] == ansCorrect[i - 2]:
            print('มีคำตอบซ้ำกันเกิน 2 ข้อ ได้แก่ข้อ', i+1, ',', i, ',', i - 1)
        if ansCorrect[i] == 'D':
            if (ansCorrect[i - 1] == 'C' and ansCorrect[i - 2] == 'B' and ansCorrect[i - 3] == 'A') :
                print('มีคำตอบเรียงกันเกิน 4 ข้อ ได้แก่ข้อ',i-2 ,',',i-1,',',i,',',i+1)
    if i >= 6:
        if ansCorrect[i] == 'A':
            if (ansCorrect[i - 1] == 'B' and ansCorrect[i - 2] == 'C' and ansCorrect[i - 3] == 'D'):
                 print('มีคำตอบเรียงกันเกิน 4 ข้อ ได้แก่ข้อ', i - 2, ',', i - 1, ',', i, ',', i + 1)


print('sum : ',sum)
# cv2.putText(answersheet, str(sum), (485, 165), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),thickness=5)
cv2.imshow('answersheet', answersheet)
cv2.putText(answersheet2, str(sum), (485, 165), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),thickness=5)
cv2.imshow('answersheet2', answersheet2)
cv2.waitKey(0)
