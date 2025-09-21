import cv2
import numpy as np
import pyscreenshot as pys
forcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', forcc, 8, (1920,1080))


while True:
    img= pys.grab()
    img_np=np.array(img)

    #frame= cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Screen', img_np)
    out.write(img_np)


    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

out.release()
cv2.destroyAllWindows()
# Load class labels that MobileNet SSD model can detect
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# Load the pre-trained MobileNet SSD model and configuration files
model_path = "models/MobileNetSSD_deploy.caffemodel"
config_path = "models/MobileNetSSD_deploy.prototxt"

# Initialize the DNN model
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Function to perform object detection
def detect_objects(frame, confidence_threshold=0.2):
    # Prepare the frame for the DNN model
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Iterate through detections and draw bounding boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            # Get the class label
            class_id = int(detections[0, 0, i, 1])
            label = CLASSES[class_id]

            # Get bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label on the frame
            label_text = f"{label}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Change 0 to your video path if using a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    output_frame = detect_objects(frame)

    # Display the output
    cv2.imshow("Object Detection", output_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
