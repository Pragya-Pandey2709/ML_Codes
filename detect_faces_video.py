from imutils.video import VideoStream //for VideoStream object that provides a convenient way to access the video stream from the camera
import numpy as np
#import argparse
import imutils //convenient image processing functions
import time
import cv2

print("[INFO] loading model...")
prototxt= "deploy.prototxt.txt"
model = "res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, model) 
//function reads a pre-trained model specified in prototxt and model files.
//prototxt specifies the network architecture, while model specifies the weights for the network.
//This model is based on the Single Shot Detector (SSD) framework and is trained to detect faces in images.


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
//VideoStream(src=0).start() starts the video stream from the default camera (0) and returns a VideoStream object.


time.sleep(2.0)
while True:
	frame = vs.read()
	//vs.read() reads a frame from the video stream
	//pre process
	//imutils.resize() resizes the frame to a width of 400 pixels to speed up the processing.
        //cv2.dnn.blobFromImage() creates a blob (binary large object) from the resized frame for input to the neural network.
        //The blob is normalized by subtracting the mean RGB values from the image pixels.
	frame = imutils.resize(frame, width=400)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	
	
	//Performs face detection
	//net.setInput(blob) sets the input blob for the neural network.
        //net.forward() runs the forward pass of the neural network to get the output detections.
        //The output is an array of detections, each containing the location and confidence score of a detected face.
	net.setInput(blob)
	detections = net.forward()
	
	//This block of code processes the output of the face detection model and draws bounding boxes and confidence scores on the frame for each detected face.
	
	
        //The for loop iterates over the detections returned by the neural network, with detections.shape[2] being the number of detected faces. 
	//For each detected face, the confidence score is extracted from the detections array using confidence = detections[0, 0, i, 2].
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		//The if statement filters out detections with low confidence scores. If the confidence score is less than 0.5, which is a threshold value set by the script, the loop moves on to the next detection using continue.
		//If the confidence score is greater than or equal to the threshold, the script proceeds to draw a bounding box around the face and write the confidence score on the frame using OpenCV functions.
		if confidence < 0.5:
			continue
		//The box variable is a 1D NumPy array containing the normalized coordinates of the bounding box for the current detection. The script multiplies this array by [w, h, w, h] to obtain the pixel coordinates of the bounding box relative to the input frame. The resulting values are then cast as 
		//integers and unpacked into (startX, startY, endX, endY).
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		
		//The text variable is a formatted string that displays the confidence score as a percentage with two decimal places.
		text = "{:.2f}%".format(confidence * 100)
		
		//The y variable is the y-coordinate of the text that will be written on the frame. If startY - 10 is greater than 10, y is set to startY - 10; otherwise, y is set to startY + 10
		y = startY - 10 if startY - 10 > 10 else startY + 10
		
		//Finally, cv2.rectangle() draws a rectangle around the detected face using the pixel coordinates of the bounding box. cv2.putText() writes the confidence score on the frame at position (startX, y) using a specific font (cv2.FONT_HERSHEY_SIMPLEX), font size (0.45), 
		//and color ((0, 0, 255)), with a thickness of 2 pixels.
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	
	//Displays the processed frame:
        //cv2.imshow() displays the processed frame with the bounding boxes and confidence scores.
        //cv2.waitKey(1) & 0xFF waits for a key press, and if the key is "q", it exits the while loop.
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
		
//Cleans up:
//cv2.destroyAllWindows() destroys all the windows created by OpenCV.
//vs.stop() stops the video stream and releases the resources.


cv2.destroyAllWindows()
vs.stop()
