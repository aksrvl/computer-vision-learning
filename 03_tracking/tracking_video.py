from ultralytics import YOLO
import cv2 as cv
import numpy as np
from math import sqrt
model = YOLO("yolo26n.pt")

cap = cv.VideoCapture('./test-video.mp4')
frame_count = 0
history = {}
kalman_filters = {} #store filters for every id
kalman_history = {} 
predictions = {}
fdes = {}
ade_predictions = {}
ades = {}

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
            track_id = detection[3]
            if track_id not in history:
                history[track_id] = []
            history[track_id].append((x_center, y_center))

            if track_id not in predictions:
                predictions[track_id] = []

            if track_id not in ade_predictions:
                ade_predictions[track_id] = []

            if track_id not in kalman_filters:
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
                kalman_filters[track_id] = kf
                kalman_history[track_id] = []

            else:
                kf = kalman_filters[track_id]
                measurement = np.array([[x_center], [y_center]], np.float32)
                state_pre = kf.predict()
                kf.correct(measurement)
                state_post = kf.statePost
                forecast = {"predictions": [], "errors": []}
                for i in range(1, 11):
                    future_x = state_post[0][0] + i*state_post[2][0]
                    future_y = state_post[1][0] + i*state_post[3][0]
                    forecast["predictions"].append((frame_count + i, future_x, future_y))
                    cv.circle(frame, (int(future_x), int(future_y)), 5, (255, 0, 0), -1)
                    if i==10:
                        target_frame = frame_count + 10
                        predictions[track_id].append((target_frame, future_x, future_y))
                ade_predictions[track_id].append(forecast)
                kalman_history[track_id].append((state_post[0][0], state_post[1][0]))

                # print(f"Predicted state {detection[0]}, ID {track_id}: ", state_pre)
                # print("Corrected state: ", state_post)
                for prediction in predictions[track_id]:
                    if prediction[0] == frame_count:
                        predicted_x = prediction[1]
                        predicted_y = prediction[2]
                        dx = predicted_x - x_center
                        dy = predicted_y - y_center
                        error = sqrt(dx**2 + dy**2)
                        if track_id not in fdes:
                            fdes[track_id] = []
                        fdes[track_id].append(error)
                        # print(f"ID {track_id} | 10-frame error: {error} px")
                        predictions[track_id].remove(prediction)                    
            # print(f"{detection[0]}| id: {track_id}| confidence: {detection[1]} | box: {detection[2]}")
                
                for forecast in ade_predictions[track_id]:
                    for prediction in forecast["predictions"]:
                        if prediction[0] == frame_count:
                            if track_id not in ades:
                                ades[track_id] = []
                            predicted_x = prediction[1]
                            predicted_y = prediction[2]
                            dx = predicted_x - x_center
                            dy = predicted_y - y_center
                            error = sqrt(dx**2 + dy**2)
                            forecast["errors"].append(error)
                            if len(forecast["errors"]) == 10:
                                ade = 0
                                for error in forecast["errors"]:
                                    ade += error
                                ade /=10
                                ades[track_id].append(ade)

                            print(f"ID {track_id} | error: {error} px")


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
total_error = 0
total_predictions = 0  
for track_id, errors in fdes.items():
            sum_fde = 0
            for i in range(0, len(errors)):
                sum_fde += errors[i]
            mean_fde = sum_fde / len(errors)
            total_error += sum_fde
            total_predictions += len(errors)
            # print(f"Mean FDE for ID {track_id} = {mean_fde}")

total_fde = total_error / total_predictions

ade_total_error = 0
ade_total_predictions = 0
for track_id, errors in ades.items():
    sum_ade = 0
    for i in range(len(errors)):
        sum_ade += errors[i]
    if len(errors) > 0:
        mean_ade = sum_ade / len(errors)
        ade_total_error += sum_ade
        ade_total_predictions += len(errors)
        print(f"Mean ADE for ID {track_id} = {mean_ade}")

total_ade = ade_total_error / ade_total_predictions
print(f"Overall Mean ADE@10: {total_ade} px")
print(f"Number of evaluated forecasts: {ade_total_predictions}")
# print(f"Overall Mean FDE@10: {total_fde} px")
# print(f"Number of evaluated predictions: {total_predictions}")
# print("History of tracked objects:")
# for track_id, positions in history.items():
#     print(f"Track ID: {track_id}, Positions: {positions}")
cap.release()
cv.destroyAllWindows()
