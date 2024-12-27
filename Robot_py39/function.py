import cv2
from collections import Counter  # Import Counter from collections module


def convert_BBxyxy_to_CWH(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    cx = int(x1 + w / 2)
    cy = int(y1 + h / 2)
    return cx, cy, w, h


def count_objects_in_image(object_classes, image):
    counter = Counter(object_classes)
    # print("Object Count in Image:")
    # print(counter)
    n = 0
    obj_count =[]
    for obj, count in counter.items():
        # print(f"{obj}: {count}")
        cv2.putText(image, f'{obj}', (50, 50 + n), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.putText(image, f'{count}', (150, 50 + n), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        n = n + 50

        # cv2.imshow("img", image)
        obj_count.append(obj)
        obj_count.append(count)

    #print(objquantity)
    return obj_count


def drawCircle_center_image(cord, image):
    cx, cy, w, h = convert_BBxyxy_to_CWH(cord[0], cord[1], cord[2], cord[3])
    cv2.circle(image, (cx, cy), 5, (255, 0, 0), 2)
    return cx,cy
