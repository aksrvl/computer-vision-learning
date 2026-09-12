import cv2 as cv
import numpy as np

kf = cv.KalmanFilter(4, 2) #4 means the overall number of variables we want to track, 2 means the number of variables we want in the output

kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) #identify which variables we want to measure
kf.transitionMatrix = np.array([
    [1, 0, 1, 0], 
    [0, 1, 0, 1], 
    [0, 0, 1, 0], 
    [0, 0, 0, 1]
    ], np.float32) #identify how the variables are related to each other
kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03 #identify how much noise we expect in the process
kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.1

# measurement = np.array([[2], [1]], np.float32) #initialize the measurement variable
# prediction = np.zeros((2, 1), np.float32) #initialize the prediction variable

measurements = [
    (2, 1), 
    (4.3, 1.8), 
    (5.7, 3.2), 
    (8.4, 3.7), 
    (9.6, 5.3)
]
kf.statePost = np.array([[2], [1], [0], [0]], np.float32) #initialize the state variable
for measurement in measurements:
    measurement = np.array([[measurement[0]], [measurement[1]]], np.float32)
    state_pre = kf.predict() #predict the next state
    kf.correct(measurement) #correct the state with the measurement
    state_post = kf.statePost #get the corrected state

    x_next = state_post[0][0] + state_post[2][0]
    y_next = state_post[1][0] + state_post[3][0]

    print("Predicted state: ", state_pre)
    print("Corrected state: ", state_post)
    print("Next x: ", x_next)
    print("Next y: ", y_next, "\n")

for _ in range(5):
    x_next += state_post[2][0]
    y_next += state_post[3][0]
    print("Next x: ", x_next)
    print("Next y: ", y_next, "\n")