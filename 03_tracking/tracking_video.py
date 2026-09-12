from ultralytics import YOLO
import cv2 as cv
import numpy as np

model = YOLO("yolo26n.pt")

cap = cv.VideoCapture('./test-video.mp4')
frame_count = 0
history = {}
kalman_filters = {} #store filters for every id
kalman_history = {} 
while cap.isOpened():
    success, frame = cap.read()
    if success:
        frame_count += 1
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf = 0.25)
        name = [result[0].names[cls.item()] for cls in result[0].boxes.cls]
        conf = result[0].boxes.conf
        box = result[0].boxes.xyxy.numpy()
        track_id = result[0].boxes.id.numpy()

        frame = result[0].plot()
        for detection in zip(name, conf, box, track_id):
            x_center = (detection[2][0] + detection[2][2])/2
            y_center = (detection[2][1] + detection[2][3])/2
            if detection[3] not in history:
                history[detection[3]] = []
            history[detection[3]].append((x_center, y_center))

            if detection[3] not in kalman_filters:
                kf = cv.KalmanFilter(4, 2)
                kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) #identify which variables we want to measure
                kf.transitionMatrix = np.array([
                    [1, 0, 1, 0], 
                    [0, 1, 0, 1], 
                    [0, 0, 1, 0], 
                    [0, 0, 0, 1]
                ], np.float32) #identify how the variables are related to each other
                kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03 #identify how much noise we expect in the process
                kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.1
                kf.statePost = np.array([[x_center], [y_center], [0], [0]], np.float32)
                kalman_filters[detection[3]] = kf
                kalman_history[detection[3]] = []

            else:
                kf = kalman_filters[detection[3]]
                measurement = np.array([[x_center], [y_center]], np.float32)
                state_pre = kf.predict()
                kf.correct(measurement)
                state_post = kf.statePost

                for i in range(1, 10):
                    future_x = state_post[0][0] + i*state_post[2][0]
                    future_y = state_post[1][0] + i*state_post[3][0]
                    cv.circle(frame, (int(future_x), int(future_y)), 5, (255, 0, 0), -1)

                kalman_history[detection[3]].append((state_post[0][0], state_post[1][0]))

                print(f"Predicted state {detection[0]}, ID {detection[3]}: ", state_pre)
                print("Corrected state: ", state_post)

            # print(f"{detection[0]}| id: {detection[3]}| confidence: {detection[1]} | box: {detection[2]}")
        for track_id, positions in history.items():
            for i in range(0, len(positions)):
                cv.circle(frame, (int(positions[i][0]), int(positions[i][1])), 5, (0, 255, 0), -1)

        for track_id, positions in kalman_history.items():
            for i in range(0, len(positions)):
                cv.circle(frame, (int(positions[i][0]), int(positions[i][1])), 5, (0, 0, 255), -1)

        cv.imshow("YOLOv8 Tracking", frame)

        if cv.waitKey(1) == ord('q'):
            break

    else:
        break

# print("History of tracked objects:")
# for track_id, positions in history.items():
#     print(f"Track ID: {track_id}, Positions: {positions}")
cap.release()
cv.destroyAllWindows()
