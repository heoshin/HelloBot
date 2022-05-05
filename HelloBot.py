import cv2
from cv2 import imshow
import numpy as np
from numpy import empty
from utils.opv import OpvModel  
import matplotlib.pyplot as plt
import time

from gtts import gTTS
import os.path
from playsound import playsound
import threading

mymodel1 = OpvModel("person-detection-retail-0013",device="CPU", fp="FP32")

def SetCameraMaxResolution(cam) -> tuple:
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
    
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    ret, frame = cam.read()
    
    if frame.shape[0:2] != (height, width):
        raise Exception((width, height), frame.shape)
    
    return (width, height)
    
def GetBoundingBoxes(predictions, image, conf = 0.6):
    canvas = image.copy()
    predictions = predictions[0][0]
    confidence = predictions[:,2]
    topresults = predictions[(confidence>conf)]
    (h,w) = canvas.shape[:2]
    boundingBoxes = topresults[:, 2:7] * np.array([1, w, h, w, h])
    boundingBoxes[:, 1:5] = boundingBoxes[:, 1:5].astype("int")

    return boundingBoxes

def DrawBoundingBoxes(boundingboxes, image, conf = 0.6):
    canvas = image.copy()

    for detection in boundingboxes[boundingboxes[:, 0] > conf]:
        (xmin, ymin, xmax, ymax) = detection[1:5].astype("int")

        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)
        cv2.putText(canvas, str(round(detection[0] * 100, 1)) + "%", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0), 2)
    cv2.putText(canvas, str(len(boundingboxes[boundingboxes[:, 0] > conf]))+" persons(s) detected", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0), 2)
    return canvas

def tts(text = ""):
    print("tts Text: " + text)
    filePath = "./tts/" + text + ".mp3"
    if not os.path.isfile(filePath):
        tts = gTTS(text=text, lang='ko')
        tts.save(filePath)
        print("tts Save: " + filePath)
    playsound(filePath)

def sayHello():
    playsound("./tts/hello.mp3")

def main():
    windowName = "HelloBot"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    camera = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
    SetCameraMaxResolution(camera)

    ret, frame = camera.read()
    camera_area = frame.shape[0] * frame.shape[1]
    detect_area_ratio = 0.15

    face = cv2.imread("./emoji/face.jpg")
    smile = cv2.imread("./emoji/smile.jpg")

    output_size = (720, 1280)
    imoji_size = (face.shape[0:2])
    cut_width = output_size[1] - imoji_size[1]

    is_smail = False
    dw_time = time.time() - 6

    while(True):
        ret, frame = camera.read(0)
        # frame = cv2.imread("./images.jpg")
        frmae_width = round(frame.shape[1] / frame.shape[0] * output_size[0])
        frame = cv2.resize(frame, dsize=(frmae_width, output_size[0]), interpolation=cv2.INTER_AREA)
        frame = frame[:, int(imoji_size[1] / 2):int(cut_width + imoji_size[1] / 2), :]

        predictions = mymodel1.Predict(frame)

        boundingBoxes = GetBoundingBoxes(predictions, frame, 0.6)
        drawFrame = DrawBoundingBoxes(boundingBoxes, frame)

        if len(boundingBoxes) >= 1:
            for box in boundingBoxes:
                (xmin, ymin, xmax, ymax) = box[1:5].astype("int")
                boxArea = (xmax - xmin) * (ymax - ymin)
                boxPer = boxArea / camera_area
                
                if boxPer > detect_area_ratio and dw_time + 7 < time.time():
                    print("Detected")
                    is_smail = True
                    threadHello = threading.Thread(target=sayHello)
                    threadHello.start()
                    dw_time = time.time()
                    break
        
        if dw_time + 4 < time.time():
            is_smail = False

        if is_smail:
            drawFrame = np.hstack((drawFrame, smile))
        else:
            drawFrame = np.hstack((drawFrame, face))

        cv2.imshow(windowName, drawFrame)

        if cv2.waitKey(1) & 0xFF == ord(' '):  # 스페이스 바가 감지되면 중지
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()