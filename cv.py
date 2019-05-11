import cv2
import numpy as np
from PIL import Image, ImageOps

#https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv citation for this function
def resize2SquareKeepingAspectRation(img, size, interpolation):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: return cv2.resize(img, (size, size), interpolation)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
  else:
    mask = np.zeros((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
  return cv2.resize(mask, (size, size), interpolation)

img = cv2.imread("IMG_8892.JPG")

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
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 255), 1, cv2.LINE_AA)
    roi = img[y+30:y+h-30, x+2:x+w-2]
    cv2.imwrite(str(comp) + '.jpg', roi)
    new_img = cv2.imread(str(comp) + '.jpg', cv2.IMREAD_GRAYSCALE)
    thresh, im_bw = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    final = cv2.bitwise_not(im_bw)
    resize_img = resize2SquareKeepingAspectRation(final, h-60, interpolation = cv2.INTER_AREA)
    dilation = cv2.dilate(resize_img,np.ones((5,5),np.uint8),iterations = 1)
    resize_img = resize2SquareKeepingAspectRation(dilation, 26, interpolation = cv2.INTER_AREA)
    constant = cv2.copyMakeBorder(resize_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0,0,0])

    cv2.imwrite(str(comp) + '.jpg', constant)




cv2.imwrite("bounded.png", img)
