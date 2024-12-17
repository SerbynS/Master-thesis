# import cv2
# import time
#
# capture = cv2.VideoCapture('http://192.168.0.64:81/stream')
# total_frames = capture.get(cv2.CAP_PROP_FPS)
# fps = 0
# while True:
#    start_time = time.time()
#    frame = capture.read()[1]
#
#    if cv2.waitKey(1) == ord('q'):
#        break
#    flipped_frame = cv2.flip(frame, 0)  # 0 - относительно вертикальной оси
#
#    cv2.putText(flipped_frame, str(f'FPS: {fps}'), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
#    cv2.imshow('frame', flipped_frame)
#    end_time = time.time()
#    fps = int(1 / (end_time - start_time))
# capture.release()
# cv2.destroyAllWindows()


import cv2

object_width = 4
distance_to_object = 15

ref_image = cv2.imread('./refer.jpg')

class_name = 'Stop sign'
stop_classifier = cv2.CascadeClassifier('./Cascade_Distance/stop_data.xml')
cv2.namedWindow("Ref image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Ref image", 320, 240)

cap = cv2.VideoCapture('http://192.168.0.64:81/stream')


def focal_length(measuredDistance, objWidthInPx, measuredWidth):
    focalDistance = (measuredDistance * objWidthInPx) / measuredWidth
    return focalDistance


def distance_counter(focalDistance, measuredWidth, objWidthInPx):
    distance = (focalDistance * measuredWidth) / objWidthInPx
    return distance


def image_process(image_frames):
    gray = cv2.cvtColor(image_frames, cv2.COLOR_BGR2GRAY)
    found = stop_classifier.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
    if len(found) != 0:
        for x, y, w, h in found:
            cv2.rectangle(image_frames, (x, y), (x + w, y + h), (255, 0, 0), 4)
            cv2.putText(image_frames, class_name, (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            return w
    return 120



object_width_in_pixels = image_process(ref_image)
print(object_width_in_pixels)
focal_distance = focal_length(distance_to_object, object_width_in_pixels, object_width)
print(focal_distance)
cv2.imshow("Ref image", ref_image)

while True:
    frame = cap.read()[1]
    object_width_in_pixels_frame = image_process(frame)
    if object_width_in_pixels_frame != 0:
        distance = distance_counter(focal_distance, object_width, object_width_in_pixels_frame)
        print('Distance:', round(distance, 2))
        cv2.putText(frame, f'Distance: {round(distance,2)} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imshow('Counter', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
