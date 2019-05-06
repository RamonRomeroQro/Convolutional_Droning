import cv2
import numpy as np

camera = cv2.VideoCapture(0)
while (True):
    return_value,image = camera.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('image',gray)
    if cv2.waitKey(1)& 0xFF == ord('s'):
        cv2.imwrite('test.jpg',image)


        ###


        img = cv2.imread('test.jpg')
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
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imwrite("cameraok.jpg", thresh)


        ##
        break
camera.release()
cv2.destroyAllWindows()