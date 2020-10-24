#calslock@github.com
import pytesseract as pyt
import cv2
import argparse
import numpy as np
import imutils
import re

parser = argparse.ArgumentParser()
parser.add_argument('-I', '--image', type=str, required=True)
args = parser.parse_args()

#Assume Tesseract is in PATH
#else uncomment next line and add path to executable
#pyt.pytesseract.tesseract_cmd = r'<add path here>'

#tesseract config
config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=ACEFGHJKLMNPQRSTUVWXY0123456789 -c language_model_penalty_non_freq_dict_word=1 -c language_model_penalty_non_dict_word=1'

#cv2 read image
img = cv2.imread(args.image)

#cv2.imshow('original', img)
#cv2.waitKey()

#change color to grey and apply bluring filter
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 15, 15, 15)

cv2.imshow('gray', gray)
cv2.waitKey()

#extract edges from image and search for biggest polygon with 4-lines
edges = cv2.Canny(gray, 30, 200)

cv2.imshow('edges', edges)
cv2.waitKey()

contoured = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contoured)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]
foundContour = None;

for i in contours:
    perimeter = cv2.arcLength(i, True)
    approxd = cv2.approxPolyDP(i, 0.018 * perimeter, True)
    if len(approxd) == 4:
        foundContour = approxd
        break

if foundContour is None:
    found = False
    print ("No licence plate detected")
else:
    found = True

if found:
    cv2.drawContours(img, [foundContour], -1, (0,0,255), 3)

    #if contour is found draw mask around it
    mask = np.zeros(gray.shape, np.uint8)
    masked = cv2.drawContours(mask, [foundContour], 0, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('masked', masked)
    cv2.waitKey()

    #and crop image
    (x,y) = np.where(mask==255)
    cropped = img[np.min(x):np.max(x)+1, np.min(y):np.max(y)+1]

    #cv2.imshow('cropped', cropped)
    #cv2.waitKey()

    ocr = pyt.image_to_string(cropped, config=config)
    ocr = ocr.replace(":", " ").replace("-", " ").replace("\n", "")
    reocr = re.search(r'([A-Z0-9]){2,3}\s{1}([A-Z0-9]){4,5}', ocr)
    if reocr is None:
        reocr = re.search(r'([A-Z0-9]){2,3}([A-Z0-9]){4,5}', ocr)
    print(reocr)