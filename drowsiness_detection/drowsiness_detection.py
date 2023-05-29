import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import matplotlib.pyplot as plt

# function to play an alarm sound
def play_alarm(sound_path):
    # Play an alarm sound
    playsound.playsound(sound_path)

# function to calculate the eye aspect ratio (EAR)
def calculate_ear(eye):
    # calculate the Euclidean distances between the two sets of vertical eye landmarks (x, y coordinates)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # calculate the Euclidean distance between the horizontal eye landmarks (x, y coordinates)
    C = dist.euclidean(eye[0], eye[3])

    # calculate the EAR
    ear = (A + B) / (2.0 * C)

    # return the EAR
    return ear

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alarm", type=int, default=0,
                help="Play alarm sound?")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="Webcam index")
args = vars(ap.parse_args())

# Constants for the EAR threshold and the number of consecutive frames the eye must be below the threshold to trigger the alarm
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 20

# Initialize the frame counter and a boolean variable to indicate if the alarm is sounding
frame_counter = 0
alarm_on = False

# initialize the dlib face detector (HOG-based) and create the facial landmarks predictor
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get the facial landmark indices for the left and right eyes, respectively
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start the video stream
print("[INFO] Starting video stream...")
vs = cv2.VideoCapture(args["webcam"])  # Change the video source to EpocCam (use 1 if necessary)
time.sleep(1.0)

# Initialize the plot for animations
y = [None] * 100
x = np.arange(0, 100)
fig = plt.figure()
ax = fig.add_subplot(111)
li, = ax.plot(x, y)

# Loop over the video stream frames
while True:
    # Read the frame from the video stream, resize it, and convert it to grayscale
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the detected faces
    for rect in rects:
        # Determine the facial landmarks for the face region, then convert the facial landmarks (x, y) coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the coordinates of the left and right eyes, then use the coordinates to calculate the EAR for both eyes
        left_eye = shape[left_eye_start:left_eye_end]
        right_eye = shape[right_eye_start:right_eye_end]
        ear_left = calculate_ear(left_eye)
        ear_right = calculate_ear(right_eye)

        # Calculate the average EAR for both eyes
        ear = (ear_left + ear_right) / 2.0

        # Calculate the convex hull for the left and right eyes, then visualize each eye
        hull_left = cv2.convexHull(left_eye)
        hull_right = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [hull_left], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [hull_right], -1, (0, 255, 0), 1)
        
        # Storage every values for y
        y_copy = y.copy()
        y_copy.append(ear)
        print(y_copy)
        
        # Remove the first element and add the calculated EAR
        y.pop(0)
        y.append(ear)

        # Update the canvas immediately
        plt.xlim([0, 100])
        plt.ylim([0, 0.4])
        ax.relim()
        ax.autoscale_view(True, True, True)
        fig.canvas.draw()
        plt.show(block=False)

        # Set the new data
        li.set_ydata(y)
        fig.canvas.draw()

        time.sleep(0.01)

        # Check if the EAR is below the blink threshold and increment the blink frame counter
        if ear < EAR_THRESHOLD:
            frame_counter += 1

            # If the eyes have been closed for a sufficient number of frames, trigger the alarm
            if frame_counter >= CONSECUTIVE_FRAMES:
                # If the alarm is not already sounding, activate it
                if not alarm_on:
                    alarm_on = True

                    # If an alarm sound file is provided, play the sound in the background using a separate thread
                    if args["alarm"] != 0:
                        t = Thread(target=play_alarm, args=("alarm.wav",))
                        t.daemon = True
                        t.start()

                # Draw an alarm on the frame
                cv2.putText(frame, "[ALERT] DROWSINESS!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Otherwise, the EAR is not below the blink threshold, so reset the frame counter and the alarm
        else:
            frame_counter = 0
            alarm_on = False

        # Draw the calculated EAR on the frame to assist with debugging and adjusting the correct EAR thresholds and frame counters
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break the loop
    if key == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
vs.release()