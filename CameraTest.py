import cv2

def SetCameraMaxResolution(cam) -> tuple:
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
    
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    ret, frame = cam.read()
    
    if frame.shape[0:2] != (height, width):
        raise Exception((width, height), frame.shape)
    
    return (width, height)
        
def main():
    cam = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
    print(SetCameraMaxResolution(cam))

    while True:
        ret, frame = cam.read()

        print(frame.shape)
        
        cv2.imshow("cam", frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):  # 스페이스 바가 감지되면 중지
            break
        
    cam.release()
    
if __name__ == '__main__':
    main()