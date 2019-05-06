import cv2 
import numpy as np

img = cv2.imread('./mytest/1.jpg',0)
cv2.bilateralFilter(img, 9, 90,16)
#binImg = np.zeros((img.shape[0], img.shape[1]), np.uint8)   
bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)

learningRate = 0
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 50  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 40
learningRate = 0

# vari
# variableslt
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyboard simulator works

fgmask = bgModel.apply(img, learningRate=learningRate)
kernel = np.ones((3, 3), np.uint8)
fgmask = cv2.erode(fgmask, kernel, iterations=1)
res = cv2.bitwise_and(img, img, mask=fgmask)



blur = cv2.GaussianBlur(res, (blurValue, blurValue), 0)
ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
cv2.imwrite("ok.jpg", thresh)

# import cv2;
# import numpy as np;
 
# # Read image
# im_in = cv2.imread("nickel.jpg", cv2.IMREAD_GRAYSCALE);
 
# # Threshold.
# # Set values equal to or above 220 to 0.
# # Set values below 220 to 255.
 
# th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);
 
# # Copy the thresholded image.
# im_floodfill = im_th.copy()
 
# # Mask used to flood filling.
# # Notice the size needs to be 2 pixels than the image.
# h, w = im_th.shape[:2]
# mask = np.zeros((h+2, w+2), np.uint8)
 
# # Floodfill from point (0, 0)
# cv2.floodFill(im_floodfill, mask, (0,0), 255);
 
# # Invert floodfilled image
# im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
# # Combine the two images to get the foreground.
# im_out = im_th | im_floodfill_inv
 
# # Display images.
# cv2.imshow("Thresholded Image", im_th)
# cv2.imshow("Floodfilled Image", im_floodfill)
# cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
# cv2.imshow("Foreground", im_out)
# cv2.waitKey(0)