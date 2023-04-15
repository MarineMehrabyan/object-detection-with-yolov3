import cv2 #use to read and display images, and perform object detection. 
import numpy as np #is a numerical computing library

def detect_objects(image_path):
    # loads the pre-trained YOLOv3 model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    # reads in the image
    img = cv2.imread(image_path)
    
    #This line extracts the height and width of the image, and discards the third value returned by the img.shape function, which corresponds to the number of color channels in the image.
    height, width, _ = img.shape
    
    #This line prepares the image for input into the YOLOv3 model. It creates a 4-dimensional blob (binary large object) from the input image, with a size of (608, 608) pixels, and normalizes the pixel values to the range of 0 to 1. It also sets swapRB=True to indicate that OpenCV should swap the red and blue channels of the image, and crop=False to indicate that the image should not be cropped. 
    blob = cv2.dnn.blobFromImage(img, 1/255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    
    #This line sets the input to the YOLOv3 model to be the prepared image blob 
    net.setInput(blob)
    
    
    #get the names of the output layers of the YOLOv3 model. These layers contain the final output of the model, which is a list of bounding boxes and associated object classes and confidence scores.
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in
        
    #runs the input image through the YOLOv3 model and gets the output of the output layers we specified earlier.
 net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)
    
    #initialize some variables
    classes = []
    confidences = []
    boxes = []
    object_count = 0
    
    #extract the bounding boxes and associated object classes and confidence scores from the output of the YOLOv3 model. It loops through each output and detection, computes the bounding box coordinates, and appends the class, confidence, and bounding box information to the respective lists
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                classes.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    #Loop over the boxes and draw a rectangle and label for each detected object
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[i])
        color = (0, 255, 0)  # green color
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    #Apply non-maxima suppression (NMS) to eliminate redundant and overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
    
    #Loop over the remaining boxes and count the number of detected objects that are not class 0 (background).
    for i in indices:
        i = i[0]
        if classes[i] != 0:
            object_count += 1
            
    #Print the total number of objects detected.
    print("Total objects detected:", object_count)

    cv2.imshow("Detected Objects", img)
    cv2.waitKey(0)



detect_objects("img1.jpg")
detect_objects("img2.jpg")
detect_objects("img3.jpg")
detect_objects("img4.jpg")
detect_objects("img5.jpg")

cv2.destroyAllWindows()
    
    
    
    
    
    
    

