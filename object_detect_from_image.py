import numpy as np
import cv2

img_to_detect = cv2.imread('Images/2.jpg')

img_height, img_width = None,None
# class_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

with open("E:\CODING PLAYGROUND\CODE\Ai-Project\YOLO\yolo-coco-data\coco.names") as f:
     class_labels = [line.strip() for line in f]









#Convert to blob to pass into model


#Recommended by yolo authors 
"""
scale factor is 0.00392 = 1/255 since 255 is the maximum pixel value of any color in an image.

By default the color factor of OpenCV is BGR instead of RGB hence we have to swap it again to get a proper image and hence to swap we write that parameter as True.

width,height of blob is 320, 320
#accepted sizes are 320X320, 416X416, 609X609, More size means more accuracy but less speed!


"""


#Declare list of colors as an array.
#Split based on comma for every slpit change the type to int
#Convert it to a numby array to apply color mask

class_colors = [ "255,255,0" ,"155,135,12","255,0,0","255,0,255","0,255,255"]
class_colors = [np.array(color.split(",")).astype("int") for color in class_colors]
class_colors = np.array(class_colors)



"""
Now , we have 80 classes to predict our output on and 5 colors and hence each color will be assigned to 16 class there for 1 color for 16 classes

Tile is a method to apply these colors to a specific class

"""
class_colors = np.tile(class_colors, (16,1))

#Loading pretrained model
#input prerocessed bolob into the model and pass throught the model
#obtain the detection predictions by the model using forward() method

yolo_model = cv2.dnn.readNetFromDarknet("E:\CODING PLAYGROUND\CODE\Deep Leaning\YOLO\model\yolov3.cfg","E:\CODING PLAYGROUND\CODE\Deep Leaning\YOLO\model\yolov3_2.weights")

#Get all the layers from the yolo network
#Loop and find the last layer (Output layer) of the yolo network
yolo_layers = yolo_model.getLayerNames()
yolo_output_layer = [yolo_layers[yolo_layer-1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]


       

if img_height is None or img_height is None:
    img_height,img_width = img_to_detect.shape[:2]


img_blob = cv2.dnn.blobFromImage(img_to_detect, 1/255, (416,416), swapRB = True, crop = False)

#Input preprocessed blob into the model and pass theough the model

yolo_model.setInput(img_blob)

#Obtain the detection layer by forwarding thorugh till the output layer


#Loop over each detection 
obj_detection_layers = yolo_model.forward(yolo_output_layer)

# _______________________________NMS_CHANGE_1_______________
#initialization for non-max supression(NMS)
#declare the following lists

class_ids_list = []
boxes_list = []
confidences_list = []
# _______________________________NMS_CHANGE_1_end_______________

    #Loop over each detection
for object_detection_layer in obj_detection_layers:
    for object_detection in object_detection_layer:
        #Structure of object detection
        # [1 to 4] =>will have the two center points box width and box height
        #[5] will have scores for all objects within  bounding boxes
        all_scores = object_detection[5:]
        predicted_class_id = np.argmax(all_scores)
        prediction_confidence = all_scores[predicted_class_id]
        if prediction_confidence >=0.5:
        #get the predicted label
            predicted_class_label = class_labels[predicted_class_id]
            print("Predicted class label", predicted_class_label)
            #obtain the bounding box co-ordunates for actual image for image size
            bounding_box = object_detection[0:4]*np.array([img_width, img_height,img_width,img_height])
            (box_center_x_pt,box_center_y_pt,box_width,box_height) = bounding_box.astype('int')
            start_x_pt = int(box_center_x_pt- (box_width/2))
            start_y_pt = int(box_center_y_pt- (box_height/2))
            
            #    ___________________________NMS_CHANGE_2_______________
            #save class id, start, x,y width & height confidences in a list for nms processing
            #make sure to pass confidence as float and width and height as integers
            
            class_ids_list.append(predicted_class_id)
            confidences_list.append(float(prediction_confidence))
            boxes_list.append([start_x_pt,start_y_pt,int(box_width),int(box_height)])
            #    ___________________________NMS_CHANGE_2_end_______________
            
            
            
            
            #____________________________NMS_CHANGE_3_________________
                
                
                #applying the NMS will retun only the selected max values ids while suppressing the non-max(weak) overlapping bounding boxes
                #non-maxim supression confidence is set as 0.5 and & max supression threshold for NMS as 0.4(Adjust and try for better perfornance)

max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list,0.5,0.3)

#Loop throught thefinal detections remaiing after NMS and draw bouding and text boxes
if len(max_value_ids)>0:
    for max_value_id in max_value_ids.flatten():
        max_class_id = max_value_id
        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]
        
        #Getting the predicted class id and label
        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = class_labels[predicted_class_id]
        prediction_confidence = confidences_list[max_class_id]
        
    #____________________________________________NMS_CHANGE_3_END____________
        
    
        end_x_pt = start_x_pt + box_width
        end_y_pt = start_y_pt + box_height
        #get a random mask color from the numpy array of the color
        box_color = class_colors[predicted_class_id]
        #convert the color numpy array into a list and apply to the textbox
        box_color = [int(c) for c in box_color]
        
        #princ the prediction
        # predicted_class_label = f"{predicted_class_label}{prediction_confidence*100}"
        predicted_class_label = f"{predicted_class_label}"
        print(f"predicted object {predicted_class_label}")

#finally drawa rectangle nad the text in the image
        cv2.rectangle(img_to_detect, (start_x_pt,start_y_pt), (end_x_pt,end_y_pt),box_color,thickness=1)
        cv2.putText(img_to_detect, predicted_class_label,(start_x_pt,start_y_pt-7), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8 , color = box_color, lineType=cv2.LINE_4)

            
            #Terminate while loop of q key is pressed

cv2.imshow("Detection Output", img_to_detect)
cv2.waitKey(0)
cv2.destroyAllWindows()