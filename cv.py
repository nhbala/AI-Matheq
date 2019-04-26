import cv2
import numpy as np
from PIL import Image, ImageOps


# def combine_rectangles(rect1, rect2):
#
#

img = cv2.imread("example.JPG")

morph = img.copy()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))

# take morphological gradient
gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

gray = cv2.cvtColor(gradient_image, cv2.COLOR_BGR2GRAY)

#take this out?
img_grey = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

# blur = cv2.medianBlur(gray,5)

blur = cv2.medianBlur(img_grey,3)


ret, thing = cv2.threshold(blur, 0.0, 255.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

img_dilation = cv2.dilate(thing, kernel, iterations=3)

cv2.imwrite("check_equal.jpg", img_dilation)

conturs_lst = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


coor_lst = []
for cnt in conturs_lst:
    x,y,w,h = cv2.boundingRect(cnt)
    if w < 15 or h < 15:
        continue
    coor_lst.append((x,y,w,h))


coor_lst.sort(key=lambda tup: tup[0])

comp = 0
for coor in coor_lst:
    comp +=1
    x = coor[0]
    y = coor[1]
    w = coor[2]
    h = coor[3]
    print(x)
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 255), 1, cv2.LINE_AA)
    roi = img[y+2:y+h-2, x+2:x+w-2]
    cv2.imwrite(str(comp) + '.jpg', roi)
    new_img = cv2.imread(str(comp) + '.jpg', cv2.IMREAD_GRAYSCALE)
    thresh, im_bw = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    resize_img = cv2.resize(im_bw,(28 , 28))
    cv2.imwrite(str(comp) + '.jpg', resize_img)




cv2.imwrite("bounded.png", img)
