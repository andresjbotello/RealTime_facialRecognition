import requests
import numpy as np
import cv2


while True:
    img_resp = requests.get("http://192.168.0.121:8080/shot.jpg")
    img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
    img = cv2.imdecode(img_arr,-1)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
