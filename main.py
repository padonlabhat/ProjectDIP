import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.measure import label,regionprops
import array as arr

def peepsctive(thresh):
    corners = cv2.goodFeaturesToTrack(thresh, 4, 0.01, 100)
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

    edged = cv2.Canny(thresh, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cv2.imwrite('output/4pre_edged.jpg', edged)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        epsilon = 0.06 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        # contourSS = cv2.drawContours(img, [approx], 0, (0, 255,0), 2)
        print(approx)
    cv2.imwrite('output/5contour.jpg', img)

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

def preimg(imS) :
    cv2.imwrite('output/1img.jpg', imS)
    gray = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/2pre_blurred.jpg', gray)

    thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.threshold(gray, 132, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite('output/3pre_thresh.jpg', thresh)
    return thresh

# ***********************************************************************************************************************
img = cv2.imread('input img/test (6).jpg')
# img = cv2.imread('input img/test (13).jpg')

# imS= cv2.resize(img, (5000, 4700))

peepsctive(preimg(img))
A4 = cv2.imread('output/6_perspective.jpg')

# plt.subplot(1,4,1)
# plt.imshow(img,cmap='gray')
# plt.subplot(1,4,2)
# plt.imshow(A4,cmap='gray')
# plt.subplot(1,4,3)
# plt.imshow(D,cmap='gray')
# preimg(A4)
# plt.show()
answersheet= cv2.resize(A4, (620,877))

# cv2.imshow('answersheet)', answersheet)
# cv2.waitKey(0)
cropped_image = img[80:280, 150:330]
gray = cv2.cvtColor(answersheet, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)[1]
# thresh = cv2.threshold(gray, 152, 255, cv2.THRESH_BINARY)[1]

# cv2.imwrite('output/thresh-A4.jpg', thresh)
# cv2.imshow('thresh-answersheet)', thresh)
# cv2.waitKey(0)

plt.imshow(thresh)
plt.show()
# crop_img = img[y:y+h, x:x+w]

y=250
h=18
w=32
ansCorrect=[]
sum =0
ansUser=['A', 'B', 'C', 'D', 'C', 'B', 'A', 'A', 'A', 'B', 'B', 'A', 'B', 'C', 'D' ,'D', 'C', 'B', 'A', 'D', 'C', 'G', 'B', 'B', 'D', 'C', 'A', 'B', 'C', 'D']
choose=''
for i in range(30):
    if i < 15 :
        x = 146

        print('loop 1-', i)
    else :
        print('loop 2-', i)
        if i == 15 :
            y=250

        x = 366
    print('ข้อที่ ', i+1)
    num = 0
    for j in range(4):
        cropped_image = thresh[y:y+h, x:x+w]
        L = label(cropped_image)
        props = regionprops(L)

        print('ตัวเลือกที่ ',j+1 ,'sum : ',props.__len__() )
        # cv2.rectangle(answersheet, (x, y), (x + w, y + h), (0, 0, 255), 1)

        if props.__len__()== 4 :
            cv2.rectangle(answersheet, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if j+1 == 1 :
                choose = 'A'
            if j+1 == 2 :
                choose = 'B'
            if j+1 == 3 :
                choose = 'C'
            if j+1 == 4 :
                choose = 'D'
            num+=1

            # if num > 1 :
            #     cv2.rectangle(answersheet, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x += 45

    # print(num)
    if num == 1:
        ansCorrect.append(choose)
    elif num >1 :
        ansCorrect.append('More')
    else :
        ansCorrect.append('NF')


    y += 26



print('ansCorrect ', ansCorrect)
print('ansUser', ansUser)
for i in range(30):
    if ansCorrect[i]==ansUser[i]:
        sum+=1
    if i >= 3:
        if ansCorrect[i] == ansCorrect[i - 1] == ansCorrect[i - 2]:
            print('มีคำตอบซ้ำกันเกิน 2 ข้อ ได้แก่ข้อ', i, ',', i - 1, ',', i - 2)
        if ansCorrect[i] == 'D':
            if (ansCorrect[i - 1] == 'C' and ansCorrect[i - 2] == 'B' and ansCorrect[i - 3] == 'A') :
                print('มีคำตอบเรียงกันเกิน 4 ข้อ ได้แก่ข้อ',i-2 ,',',i-1,',',i,',',i+1)
    if i >= 6:
        if ansCorrect[i] == 'A':
            if (ansCorrect[i - 1] == 'B' and ansCorrect[i - 2] == 'C' and ansCorrect[i - 3] == 'D'):
                 print('มีคำตอบเรียงกันเกิน 4 ข้อ ได้แก่ข้อ', i - 2, ',', i - 1, ',', i, ',', i + 1)
print('sum : ',sum)
cv2.putText(answersheet, str(sum), (485, 165), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),thickness=5)
# cv2.imshow('cropped_image', cropped_image)
# cv2.waitKey(0)
cv2.imshow('answersheet', answersheet)
cv2.waitKey(0)
# cv2.imshow("Scanned", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()