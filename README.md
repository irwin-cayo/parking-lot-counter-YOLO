# parking-lot-counter-YOLO

Vehicle detection and tracking using YOLO object detector and dlib correlation tracking for the east entrance of parking structure PK-E at CSUSB.
Tested on Opencv-Python 4.x.x

## Getting Started

Make sure you have Git installed on your machine. Naviagte to the desired project folder and clone the repository or download the zip folder.

```
git clone https://github.com/irwin-cayo/parking-lot-counter-YOLO.git
```

Now that you have the repository on your local machine, it is best practice to create a virtual environment to house the dependencies for this project. You may use an environment creator of your choice such as virtualenv, pipenv or venv. If you do not have any of these simply install one via command line. Note: You will need to have python3 installed on your machine.

```
pip install virtualenv
```
After you have installed virtualenv, create a python virtual environment at the preffered project location:

```
virtualenv count_project
```
To activate the virtual environment:

```
C:\path\to\folder\count\Scripts\activate
```
To deactivate:

```
deactivate
```

### Prerequisites

The training script will utilize many packages such as: 

```
OpenCV-Python; dlib; numpy; etc;
```

### Installing
If you do not have these packages installed you can easily pip install them. They are listed in the requirements.txt file. To install all dependencies simply type into the command line of your python virtual environment. Note: Dlib must be compiled. This can be done with pip install but you must have a c++ compiler installed on your machine. Having visual studio installed on your machine will work as well.

```
pip install requirements.txt
```

Check to see if opencv-python is properly installed. In the python shell type:

```
import cv2
```

Then,
```
print(cv2.__version__)
```

## Running the program

Run the program on a segment of video from the UPD. Use the East entrance for this script.

In your command line type:

```
python vehicle_counter_yolo_east.py --input videos/exit_east.avi --output output/insert_name_of_output.avi --yolo yolo-coco
```

In the output folder you should now see a new video with annotations. Verify that the contents are correct.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
