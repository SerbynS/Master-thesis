import cv2
import numpy as np

net = cv2.dnn.readNet('res/yolov3-tiny.weights', 'res/yolov3-tiny.cfg')
classes = []
with open('res/coco.names') as f:
    for name in f:
        classes.append(name[:-1])

cap = cv2.VideoCapture('http://192.168.0.64:81/stream')
# cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
colors = np.random.uniform(0, 255, size=(100, 3))

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([90, 255, 255])
lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([35, 255, 255])

# lower_yellow = np.array([15, 150, 150])
# upper_yellow = np.array([35, 255, 255])

prev_color = ''


def auto_steering(traffic_color):
    if traffic_color == 'RED':
        action = 'STOP'
    elif traffic_color == 'YELLOW':
        action = 'SLOW_SPEED'
    elif traffic_color == 'GREEN':
        action = 'NORMAL_SPEED'
    else:
        action = 'SLOW_SPEED'
    print(traffic_color, action)


def detect_circles(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv2.add(mask1, mask2)
    size = img.shape
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80, param1=50, param2=10, minRadius=0, maxRadius=30)
    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60, param1=50, param2=10, minRadius=0, maxRadius=30)
    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=5, minRadius=0, maxRadius=30)
    bound = 0.4
    traffic_color = ''
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))
        for i in r_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                continue
            cv2.circle(img, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
            cv2.circle(maskr, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
            cv2.putText(img, 'RED', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            traffic_color = 'RED'

    if y_circles is not None:
        y_circles = np.uint16(np.around(y_circles))
        for i in y_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                continue
            cv2.circle(img, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
            cv2.circle(masky, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
            cv2.putText(img, 'YELLOW', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            traffic_color = 'YELLOW'

    if g_circles is not None:
        g_circles = np.uint16(np.around(g_circles))
        for i in g_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                continue
            cv2.circle(img, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
            cv2.circle(maskg, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
            cv2.putText(img, 'GREEN', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            traffic_color = 'GREEN'
    return img, traffic_color


while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2 and class_id == 9:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.1)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            crop = frame[y:y + h, x: x + w]
            frame_with_light, color_light = detect_circles(crop)

            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 1, (255, 255, 255), 2)

            if color_light != prev_color:
                auto_steering(color_light)
            prev_color = color_light

            cv2.imshow('Crop', crop)
    cv2.imshow('Traffic light', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


