import cv2

URL_STREAM = 'rtsp://172.20.10.8:8080/h264.sdp'
cap = cv2.VideoCapture(URL_STREAM, cv2.CAP_FFMPEG)

if cap.isOpened():
    print("✅ Conexión exitosa")
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Test', frame)
        cv2.waitKey(0)
else:
    print("❌ Error de conexión")
cap.release()
cv2.destroyAllWindows()