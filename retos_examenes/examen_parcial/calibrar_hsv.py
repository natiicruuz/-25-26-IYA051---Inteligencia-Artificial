import cv2
import numpy as np

def nothing(x):
    pass

URL_STREAM = 'rtsp://172.20.10.8:8080/h264.sdp'
cap = cv2.VideoCapture(URL_STREAM, cv2.CAP_FFMPEG)

cv2.namedWindow('Calibracion HSV')
cv2.createTrackbar('H_min', 'Calibracion HSV', 0, 179, nothing) 
cv2.createTrackbar('H_max', 'Calibracion HSV', 143, 179, nothing) 
cv2.createTrackbar('S_min', 'Calibracion HSV', 169, 255, nothing) 
cv2.createTrackbar('S_max', 'Calibracion HSV', 255, 255, nothing)
cv2.createTrackbar('V_min', 'Calibracion HSV', 0, 255, nothing)
cv2.createTrackbar('V_max', 'Calibracion HSV', 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    h_min = cv2.getTrackbarPos('H_min', 'Calibracion HSV')
    h_max = cv2.getTrackbarPos('H_max', 'Calibracion HSV')
    s_min = cv2.getTrackbarPos('S_min', 'Calibracion HSV')
    s_max = cv2.getTrackbarPos('S_max', 'Calibracion HSV')
    v_min = cv2.getTrackbarPos('V_min', 'Calibracion HSV')
    v_max = cv2.getTrackbarPos('V_max', 'Calibracion HSV')
    
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow('Original', frame)
    cv2.imshow('Mascara', mask)
    cv2.imshow('Resultado', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Valores óptimos:")
        print(f"LOWER_COLOR_FONDO = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"UPPER_COLOR_FONDO = np.array([{h_max}, {s_max}, {v_max}])")
        break

cap.release()
cv2.destroyAllWindows()