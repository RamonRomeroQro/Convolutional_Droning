import cv2
import numpy as np


import cv2
from matplotlib import pyplot as plt



from keras.models import load_model
import argparse
import pickle
import cv2
 
# construct the argument parser and parse the arguments
def predict2(filename, model, lb):
    
   
    img2 = np.zeros((filename.shape[0], filename.shape[0], 3))
    img2[:,:,0] = filename
    img2[:,:,1] = filename
    img2[:,:,2] = filename


    image = img2
    
    output = image.copy()
    image = cv2.resize(image, (64, 64))

    # scale the pixel values to [0, 1]
    image = image.astype("float") / 255.0

    # check to see if we should flatten the image and add a batch
    # dimension
   
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # load the model and label binarizer
   
    # make a prediction on the image
    preds = model.predict(image)

    # find the class label index with the largest corresponding
    # probability
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]

    # draw the class label + probability on the output image
    #print("{}: {:.2f}%".format(label, preds[0][i] * 100))
    return label
 
def sketch_transform(image, model, lb):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray scale

    

    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (7,7), 0) # gaussian blur
    #image_canny = cv2.Canny(image_grayscale_blurred, 10, 80)
    _, mask = image_canny_inverted = cv2.threshold(image_grayscale, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = mask
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
    fgmask = cv2.erode(fgmask, kernel, iterations=100)
    res = cv2.bitwise_and(img, img, mask=fgmask)


    ret, thresh = cv2.threshold(res, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # filename= ".current.jpg"
    # cv2.imwrite( filename, thresh)

    final = cv2.bitwise_not(thresh)
    

    label = predict2(final, model, lb)
    


    return final, label






from pyparrot.Bebop import Bebop

bebop = Bebop(drone_type="Bebop2")
print("CONNECTING...")
success = bebop.connect(10)
print(success)
if success:

    # No cambiar los valores de las siguientes 3 configuraciones !!!!!!
    bebop.set_max_altitude(2)           #Establece la altitud máxima en 3 metros
    bebop.set_max_tilt(10)               #Establece la velocidad máxima de movimientos angulares en 5°
    bebop.set_max_vertical_speed(0.5)   #Establece la velocidad máxima vertical en 0.5 m/s



    
cam_capture = cv2.VideoCapture(0)
cv2.destroyAllWindows()
upper_left = (50, 50)
bottom_right = (300, 300)

print("[INFO] loading network and label binarizer...")

model = load_model("output/cnn.model")
lb = pickle.loads(open("output/cnn.pickle", "rb").read())

while True:
    _, image_frame = cam_capture.read()
    
    #Rectangle marker
    r = cv2.rectangle(image_frame, upper_left, bottom_right, (100, 50, 200), 5)
    rect_img = image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    
    sketcher_rect = rect_img
    sketcher_rect, label = sketch_transform(sketcher_rect, model, lb)
    
    #Conversion for 3 channels to put back on original image (streaming)
    sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)
    
    #Replacing the sketched image on Region of Interest
    image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]] = sketcher_rect_rgb
    cv2.imshow("Sketcher ROI", image_frame)
    
    #print(label)

    if label=="ok": 
        print('taking off')
        #bebop.safe_takeoff(10)
    elif label=="C": 
        print('forward')
        #bebop.fly_direct(0, 10, 0, 0, 0.1)
    elif label=="L": 
        print('backward')
        #bebop.fly_direct(0, -10, 0, 0, 0.1)
    elif label=="fist": 
        print('left')
        #bebop.fly_direct(-10, 0, 0, 0, 0.1)
    elif label=="okay": 
        print('right')
        #bebop.fly_direct(10, 0, 0, 0, 0.1)
    elif label=="palm": 
        print('up')
        #bebop.fly_direct(0, 0, 0,10, 0.1)
    elif label=="peace": 
        print('up')
        #bebop.fly_direct(0, 0, 0,10, 0.1)
    

    # p = Popen(["python3", "scream.py"], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    # p.stdin.write(str(label).encode('utf-8'))
    # p.stdin.flush()
    # #p.stdin.close()
    # p.wait()
    

    #stdout_data = p.communicate(label.encode())[0]
    if cv2.waitKey(1)& 0xFF == ord('s'):
        break

cam_capture.release()
cv2.destroyAllWindows()