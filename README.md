_____Object Detection using YOLOv3_____

This code uses the YOLOv3 (You Only Look Once version 3) model to detect objects in images. The YOLOv3 model is a state-of-the-art deep learning model that can perform object detection in real-time.

_____Dependencies_____
This code uses the following Python libraries:

cv2: This library is used to read and display images, and perform object detection.
numpy: This is a numerical computing library.


_____Usage_____

To run this code, you need to have the following files in the same directory as the code:

yolov3.weights: This is the pre-trained weights file for the YOLOv3 model.
yolov3.cfg: This is the configuration file for the YOLOv3 model.

You can call the detect_objects() function with the path of an image as an argument to detect objects in that image. The function will display the image with the detected objects highlighted by bounding boxes and labels. It will also print the total number of objects detected in the image.

You can also call the detect_objects() function with multiple images to detect objects in all of them.
