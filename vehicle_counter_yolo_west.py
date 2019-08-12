# Read and write back out to video:
# python vehicle_counter_yolo_west.py --input videos/name_of_input_video.avi --output output/name_of_output.avi --yolo yolo-coco

# To read from webcam and write back out to disk:
# python vehicle_counter_yolo_west.py --output/webcam_output.avi --yolo yolo-coco

# Import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from datetime import datetime
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

# Construct the argument parse and parse the arguments entered in the command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")

# Each detection will have a confidence level. We will filter out weak detections.
# Low confidence detections will be ignored. 
ap.add_argument("-c", "--confidence", type=float, default=0.75,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
# Object detection will only be performed every nth frame
ap.add_argument("-s", "--skip-frames", type=int, default=15,
	help="# of skip frames between detections")
args = vars(ap.parse_args())


# This script uses Yolo-v3 pre-trained model by Darknet and was 
# Trained on the COCO (common objects in context) dataset
# The COCO dataset contains 80 labels. Yolo-v3 object detector can classify these 80 different objects.
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Create pseudo random list of random colors 
# We will use these for the 80 different labels/objects. But really we will only use 3 of these colors.
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# We do not need to train our own model for vehicle detection since there are already many
# pre-trained models that work for general vehicle detection.
# Since the model we are using is pre-trained we only need the weights and configuration
# located in the 'yolo' folder in the directory.

# Derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Load Yolo-v3 and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# If an input video path was not supplied, grab a reference to the webcam!
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# Otherwise, grab a reference to the video file that will be used
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# Initialize the video writer for the output video
writer = None

# initialize frame dimensions
W = None
H = None

file_object = open("cv_detection_results_west.txt", "a+")

# Instantiate our centroid tracker, then initialize a list to store each of 
# our dlib correlation trackers, followed by a dictionary to map TrackableObject
ct = CentroidTracker(maxDisappeared=25, maxDistance=150) 
trackers = []
trackableObjects = {}

local_time = 1554215400.481
time = datetime.fromtimestamp(local_time)

# initialize the total number of frames processed 
# and total vehicles who have entered and exited
totalFrames = 0
totalDown = 0
totalUp = 0
temp_in = 0
temp_out = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over each frame from the video file or stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# if we are reading a video and didn't grab a frame then we reached the end of video
	if args["input"] is not None and frame is None:
		break

	# resize the frame to have a maximum width of 1080 pixels
	# yolo will use BGR frame, but dlib requires RGB frame so create copy of frame in RGB
	frame = imutils.resize(frame, width = 1080)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# initialize writer if we are going to be writing to disk
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W,H), True)

	# initialize the current status (waiting, detecting, tracking) 
	# along with our list of bounding box rectangles returned by either 
	# (1) our object detector or
	# (2) the correlation trackers
	
	status = "Waiting"
	rects = []

	# Check to see if we should use object detection to detect new objects
	# or if we should track our already detected objects (if any)
	# If condition is met we use object detection
	if totalFrames % args["skip_frames"] == 0:
		# set the status and initialize our new set of object trackers
		status = "Detecting"
		trackers = []

		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# initialize list of detected bounding boxes, confidences, and classID's
		boxes = []
		confidences = []
		classIDs = []

		# loop over the detections
		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability)
				# of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				
				# filter out weak predictions
				if confidence > args["confidence"]:
					if LABELS[classID] in ["car","truck", "motorbike"]:

						# scale the bounding box coordinates back relative to
						# the size of the image, keeping in mind that YOLO
						# actually returns the center (x, y)-coordinates of
						# the bounding box followed by the boxes' width and
						# height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top
						# and and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates,
						# confidences, and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
					else:
						continue

		# apply non-maxima suppression to suppress weak, overlapping bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
			args["threshold"])

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(x, y, x + w, y + h)
				tracker.start_track(rgb, rect)

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)

	# otherwise, we should utilize our object tracking rather than
	# object detecting to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"

			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))

	# Initialize position points on the frame which we will use for drawing lines on the frame
	# Diag lines (y) point values
	diag_start_y = (H // 2) - 188 
	diag_end_y =  (H //2 ) - 188
	# Diag Lines (x) point values
	diag_start_x = 349
	diag_end_x = (W // 2) + 60
	# Vertical Line entrance point values
	horz1 = (H // 2) + 59
	exit_x1 = (W // 2) + 115
	exit_x2 = exit_x1 + 280
	top = 270
	
	#Horizontal line
	cv2.line(frame, (diag_start_x, diag_start_y ), (diag_end_x, diag_end_y), (0, 255, 0), 1) #entrance
    
	#Vertical lines for entrance
	cv2.line(frame, (diag_end_x, diag_end_y), (diag_end_x, 50), (0, 255, 0), 1) #right line
	cv2.line(frame, (diag_start_x, diag_start_y), (diag_start_x, 100), (0, 255, 0), 1) #left line
    
	#Horizontal line for exit
	cv2.line(frame, (exit_x1, diag_end_y ), (exit_x2, diag_end_y), (0, 0, 255), 1) #exit

	#perhaps add vertical line for exit

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			if not to.entered and not to.already_in_lot:
					if (centroid[1] < diag_start_y and centroid[0] < diag_end_x) and centroid[0] > diag_start_x:
							to.entered = True
					elif (centroid[1] > diag_end_y or centroid[0] < diag_start_x): 
							to.already_in_lot = True
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			# check to see if the object has been counted yet or not
			# and location of vehicle when detected
			if not to.counted:
					# if the direction is negative (indicating the object
					# is moving up) AND the centroid is in the exit zone and was 
					# detected outside the exit zone first, then count it
					if to.already_in_lot and direction < 0:
							if ((centroid[1] < diag_end_y and centroid[0] < exit_x2)and centroid[0] > exit_x1):
									if (totalFrames - temp_in > 30):
											temp_in = totalFrames
											time = datetime.fromtimestamp(local_time)
											print("time: {} | status: -1".format(time))
											file_object.write("%s\tEast-Exit\t-1\n" % time)
											totalUp += 1
											to.counted = True
									else:
											to.counted = True
					# if the direction is positive (indicating the object
					# is moving down) AND the centroid is below the
					# center line, and was detected at entrance, count the object
					elif to.entered:
							if (centroid[0] < diag_start_x or centroid[1] > diag_start_y) and (direction > 0):
									if (totalFrames - temp_out > 30):
											temp_out = totalFrames
											time = datetime.fromtimestamp(local_time)
											print("time: {} | status: 1".format(time))
											file_object.write("%s\tEast-Entrance\t1\n" % time)
											totalDown += 1
											to.counted = True
									else:
											to.counted = True
		
		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		#text = "Vehicle {}".format(objectID)
		text = ""
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 4)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
		#cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 5)
                

    # construct a tuple of information we will be displaying on the
	# frame
	info = [
		("Exit", totalUp),
		("Enter", totalDown),
		("Status", status),
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 30) + 30)),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	local_time = 1554215400.481 + ((1000.0 * totalFrames) / 30000)
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

#close file writer
file_object.close()

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()