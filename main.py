import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from  skimage.measure import label,regionprops
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
    cv2.imwrite('output/0img.jpg', imS)
    gray = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/2pre_blurred.jpg', gray)
    thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('output/3pre_thresh.jpg', thresh)
    return thresh
# ***********************************************************************************************************************
img = cv2.imread('input img/test (6).jpg')
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
thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]


# cv2.imshow('thresh-answersheet)', thresh)
# cv2.waitKey(0)
cv2.imwrite('output/thresh-A4.jpg', thresh)

plt.imshow(thresh)
plt.show()
# crop_img = img[y:y+h, x:x+w]
y=248
x=145
h=20
w=35
ans=[]
sum =0
ans2=['A', 'C', 'A', 'A', 'D', 'C', 'G', 'B', 'B', 'D', 'C', 'A', 'A', 'B', 'D']
choose=''
for i in range(15):
    # print('ข้อที่ ', i+1)
    num = 0
    for i in range(4):
        cropped_image = thresh[y:y+h, x:x+w]
        L = label(cropped_image)
        props = regionprops(L)

        # print('ตัวเลือกที่ ',i+1 ,'sum : ',props.__len__() )
        # cv2.rectangle(answersheet, (x, y), (x + w, y + h), (0, 0, 255), 1)

        if props.__len__()== 4 :
            cv2.rectangle(answersheet, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if i+1 == 1 :
                choose = 'A'

            if i+1 == 2 :
                choose = 'B'

            if i+1 == 3 :
                choose = 'C'

            if i+1 == 4 :
                choose = 'D'

            num+=1



            # if num > 1 :
            #     cv2.rectangle(answersheet, (x, y), (x + w, y + h), (0, 255, 0), 2)


        x += 45
    # print(num)
    if num == 1:
        ans.append(choose)
    elif num >1 :
        ans.append('More')
    else :
        ans.append('NF')
    x = 145
    y += 26

print('ans ', ans)
print('ans2', ans2)
for i in range(15):
    if ans[i]==ans2[i]:
        sum+=1

print('sum : ',sum)
# cv2.imshow('cropped_image', cropped_image)
# cv2.waitKey(0)
cv2.imshow('answersheet', answersheet)
cv2.waitKey(0)
# cv2.imshow("Scanned", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
