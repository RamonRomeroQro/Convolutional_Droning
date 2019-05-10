python3 predict.py --image "$1" --model output/cnn.model --label-bin output/cnn.pickle --width 64 --height 64



# import the necessary packages
from keras.models import load_model
import argparse
import pickle
import cv2
 
# construct the argument parser and parse the arguments

image= cv2.imread(".current.jpg")
#  image=cv2.imread(args["image"]) correct

output = image.copy()
image = cv2.resize(image, (64, 64))
 
# scale the pixel values to [0, 1]
image = image.astype("float") / 255.0

# check to see if we should flatten the image and add a batch
# dimension
if args["flatten"] > 0:
	image = image.flatten()
	image = image.reshape((1, image.shape[0]))
 
# otherwise, we must be working with a CNN -- don't flatten the
# image, simply add the batch dimension
else:
	image = image.reshape((1, image.shape[0], image.shape[1],
		image.shape[2]))

# load the model and label binarizer
print("[INFO] loading network and label binarizer...")

model = load_model("output/cnn.model")
lb = pickle.loads(open("output/cnn.pickle", "rb").read())

# make a prediction on the image
preds = model.predict(image)
 
# find the class label index with the largest corresponding
# probability
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# draw the class label + probability on the output image
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(0, 0, 255), 2)
 
# show the output image
cv2.imshow("Image", output)
cv2.waitKey(0)





