import cv2
import numpy as np
from PIL import Image, ImageOps



img = cv2.imread("export.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

morphed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, np.ones((2,2)))

conturs_lst = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]


coor_lst = []
for cnt in conturs_lst:
    x,y,w,h = cv2.boundingRect(cnt)
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
    new_img = Image.open(str(comp) + '.jpg')
    bordered_image = ImageOps.expand(new_img, border=100, fill='white')
    bordered_image.save(str(comp) + '.jpg')





cv2.imwrite("bounded.png", img)
