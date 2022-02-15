import imutils
import cv2
import numpy as np

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
    h = 3500;
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



# cv2.imshow("Scanned", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
