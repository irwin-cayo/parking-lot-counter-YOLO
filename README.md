Vehicle detection and tracking using YOLO object detector and dlib correlation tracking for the east entrance of PK-E at CSUSB.
Tested on Opencv-Python 4.x.x 

Dependencies: 
- Opencv-Python
- imutils 
- dlib
- NumPy
- SciPy
- yolov3

To read and write back out to video (in command line):
python vehicle_counter_yolo.py --input videos/PKEE.avi --output output/PKEE1output_01.avi --yolo yolo-coco

To read from webcam:
python vehicle_counter_yolo.py --yolo yolo-coco
