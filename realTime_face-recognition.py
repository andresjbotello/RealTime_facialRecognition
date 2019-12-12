import requests
import matplotlib.pyplot as plt
import numpy as np
import cvlib as cv
import cv2

def main():
    while True:
        img_resp = requests.get("http://192.168.0.121:8080/shot.jpg")
        img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
        img = cv2.imdecode(img_arr,-1)

        #cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # apply face detection
        face, confidence = cv.detect_face(img)

        print(face)
        print(confidence)

        # loop through detected faces
        for idx, f in enumerate(face):
            
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)

            text = "{:.2f}%".format(confidence[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write confidence percentage on top of face rectangle
            cv2.putText(img, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0), 2)

        # display output
        cv2.imshow("Real-time face detection", img)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=='__main__':
    main()