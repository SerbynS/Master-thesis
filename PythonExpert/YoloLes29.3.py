# Нахождение всех объектов на камере, через Yolo

# указываем имя объекта, которые мы хотим искать на картинке
name_id = 11 # Отсчёт идёт с 0, 11-это знак стоп, 9-светофор

object_H1 = 3.5
object_pix_H2 = 40
distance_to_object_S1 = 9.5

def S2(s1, h1, h2):
   s2= (s1 * h2) / h1
   return s2


def distance_counter_S3(h1, s2, h4):
   distance_s3 = (h1 * s2) / h4
   return distance_s3


s2 = S2(distance_to_object_S1, object_H1, object_pix_H2)
#---------------------------------

import cv2
import numpy as np

# Выбор модели Yolo. Ссылка описывающая доступные модели Yolo - https://pjreddie.com/darknet/yolo/
net = cv2.dnn.readNet('./res/yolov2-tiny.weights', './res/yolov2-tiny.cfg')  # самая быстрая, но не точная
#net = cv2.dnn.readNet('./res/yolov3-tiny.weights', './res/yolov3-tiny.cfg') # быстрая, средняя точность
#net = cv2.dnn.readNet('./res/yolov3.weights', './res/yolov3.cfg')           # медленная, но высокая точность, при GPU быстро обрабатывает данные

classes = [] # готовим список имён, на которые обучена сеть

# чтение данных с файла и запись их в массив
with open('./res/coco.names') as f:
    for name in f:
        classes.append(name[:-1])


cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN

# выбираем случайный цвет, для создания контуров вокруг объектов
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    img = cap.read()[1]
    cv2.resize(img,(224,224)) # Устанавливает размер кадра 416 на 416, чтобы лучше обрабатывать изображение, можно и другие значения поставить 224×224, 227×227, 299×299
    height, width, _ = img.shape

    # готовит блобы, входые данные для нейронки
    # blobFromImage(image, scale, size, mean, swapRB, crop)
    # mean - значение каналов RGB которые нужно отнять у изображения, если это нужно
    # scale - в зависимости от обучения модели ее масштабирования может быть разным
    # size – есть несколько вариантов размера изображения попадающего на вход нейросети – 224×224, 227×227, 299×299 или 416×416. Чем больше изображение тем труднее работать, но точность будет выше
    # swapRB - надо делать потому что у нас начальный канал BGR, а нейросеть должна получить RGB
    # crop – вырезает часть изображения с которым работает

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (224, 224), (0, 0, 0), swapRB=True, crop=False)
    # загружает данные в нейронку
    net.setInput(blob)

    # находит все доступные объекты на изображение
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []       # набор данных x,y,w,h объекта, который найден на картинке
    confidences = [] # набор данных по каждому объекту, % на сколько этот объект правильно найден
    class_ids = []   # id бъекта в Yolo

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # если вероятность верного объекта выше 20%, то мы его записываем
            if confidence > 0.2 and class_id == name_id:

                # значения хранятся в %, мы их переводим в пиксели
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # добавляем в конец списка данные объектов, которые были определены на картинке
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)



    # Проверяем, чтобы у нас на картинке хотябы был один объект, если его нет, то мы не будем отоброжать объекты
    if len(boxes) > 0:
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4).flatten()

        # если есть элементы, то выводим их значения на экран
        if len(indexes) > 0:
            for i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y + 20), font, 1, (255, 255, 255), 2)

                print("Distance: " + str(distance_counter_S3(object_H1,s2,h)))

    # выводим изображение
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
