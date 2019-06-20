# Read and write back out to video:
# python vehicle_counter_yolo.py --input videos/PKEE.avi --output output/PKEE1output_01.avi --yolo yolo-coco
#
# To read from webcam and write back out to disk:
# python vehicle_counter_yolo.py --output/webcam_output.avi --yolo yolo-coco

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
ap.add_argument("-s", "--skip-frames", type=int, default=15,
	help="# of skip frames between detections")
args = vars(ap.parse_args())


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# initialize the video writer
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=25, maxDistance=150) #was 25 and 150
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
temp_in = 0
temp_out = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if args["input"] is not None and frame is None:
		break

	# resize the frame to have a maximum width of 1080 pixels
	#(change to dimensions as needed)
	#, then convert
	# the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width = 1080)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W,H), True)

	# initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
	status = "Waiting"
	rects = []

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % args["skip_frames"] == 0:
		# set the status and initialize our new set of object trackers
		status = "Detecting"
		trackers = []

		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections
		# blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		# net.setInput(blob)
		# detections = net.forward()

		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# initialize our lists of detected bounding boxes, confidences,
		# and class IDs, respectively
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
				
				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
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

		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
			args["threshold"])

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# draw a bounding box rectangle and label on the frame
				#THIS WILL probably give you seizures if sensitive to flashing lights
				# color = [int(c) for c in COLORS[classIDs[i]]]
				# cv2.rectangle(frame, (x, y), (x + w, y + h), color, 12)
				# text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				# 	confidences[i])
				# cv2.putText(frame, text, (x, y - 5),
				# 	cv2.FONT_HERSHEY_SIMPLEX, 2, color, 9)
				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(x, y, x + w, y + h)
				tracker.start_track(rgb, rect)

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)
					
	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing throughput
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

	# draw a horizontal line in the center of the frame -- once an
	# object crosses this line we will determine whether they were
	# moving 'up' or 'down'
	horz1 = (H // 2) + 45
	mid_horz = horz1 - 30
	vert1_ent = (W // 2) - 59
	vert2_ent = vert1_ent - 230
	vert1_ext = (W // 2) + 250
	vert2_ext = (W // 2) + 50
	top = 270
	#Horizontal lines
	cv2.line(frame, (vert2_ent, horz1 ), (vert1_ent, horz1), (0, 255, 0), 1) #entrance
	cv2.line(frame, (vert2_ext, horz1 ), (vert1_ext, horz1), (0, 0, 255), 1) #exit
	cv2.line(frame, (vert1_ent, mid_horz ), (vert2_ext, mid_horz), (0, 255, 255), 1) #edge case
        #Vertical lines for entrance
	cv2.line(frame, (vert1_ent, top), (vert1_ent, horz1), (0, 255, 0), 1) #right line
	cv2.line(frame, (vert2_ent, top), (vert2_ent, horz1), (0, 255, 0), 1) #left line
	#vertical lines for exit
	cv2.line(frame, (vert1_ext, top), (vert1_ext, horz1), (0, 0, 255), 1) #left line
	cv2.line(frame, (vert2_ext, top), (vert2_ext, horz1), (0, 0, 255), 1) #right line
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
                                if ((centroid[1] > horz1 or centroid[0] > vert1_ext) or ((centroid[0] < vert2_ext and centroid[0] > vert1_ent) and centroid[1] > mid_horz )): #if below or to the right of far right red
                                        to.already_in_lot = True
                                elif centroid[1] < horz1 and centroid[0] < vert1_ent:
                                        to.entered = True
                        #y = [c[1] for c in to.centroids]        ##check difference##################
                        #direction = centroid[1] - np.mean(y)
                        #to.centroids.append(centroid)

			# check to see if the object has been counted or not
			#and location of vehicle when detected
                        if not to.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line AND to the left of the red vettical line,
				# AND the vehicle was already in the lot, count the object
                                if to.already_in_lot: #directiom < 0 and underneath
                                        if (centroid[1] < horz1 and centroid[0] < vert1_ext):
                                                if (totalFrames - temp_in > 30):
                                                        temp_in = totalFrames
                                                        totalUp += 1
                                                        to.counted = True
                                                else:
                                                        to.counted = True
                                # if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, and was detected at entrance, count the object
                                elif to.entered:
                                        if ((centroid[0] > vert1_ent or centroid[0] < vert2_ent)
                                       or centroid[1] > horz1): #and (direction > 0):
                                                if (totalFrames - temp_out > 30):
                                                        temp_out = totalFrames
                                                        totalDown += 1
                                                        to.counted = True
                                                else:
                                                        to.counted = True

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		##GOT RIDDDD
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
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

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
