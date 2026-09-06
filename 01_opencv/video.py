##
# Відкрити відеофайл.
# Зчитувати його кадр за кадром.
# Відображати кадри у вікні.
# Завершувати роботу, коли відео закінчилося.
# Дозволяти користувачу вручну зупинити відтворення.
# Вивести в термінал:
# resolution відео;
# FPS;
# номер поточного кадру.
# На кожному кадрі намалювати прямокутник у заданих тобою координатах.


import cv2 as cv
cap = cv.VideoCapture('./01_opencv/video.mp4')
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv.CAP_PROP_FPS)
print(f"Resolution: {width}x{height}, FPS: {fps}")
frame_count = 0
e1 = cv.getTickCount()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame_count += 1
    cv.rectangle(frame, (1180, 620), (1280, 720), (255, 0, 0), 3)
    cv.rectangle(frame, (0, 0), (100, 100), (0, 255, 0), -1)
    cv.rectangle(frame, (590, 310), (690, 410), (0, 0, 255), 3)
    cv.circle(frame, (640, 360), 1, (255, 255, 0), -1)
    cv.circle(frame, (360, 360), 1, (255, 255, 0), -1)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
e2 = cv.getTickCount()
time = (e2 - e1) / cv.getTickFrequency()
print("FPS processed: ", frame_count / time)
cap.release()
cv.destroyAllWindows()
