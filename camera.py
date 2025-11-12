import cv2

#? Opens Webcam
cap = cv2.VideoCapture(0)

while True:
    #? Grabs a frame from the camera
    ret, frame = cap.read()
    #? Shows frame in a window
    cv2.imshow("Webcam Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
