import cv2
import dlib
import numpy as np

#load the pre-trained drowsiness detection model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#set up video capture
cap = cv2.VideoCapture(0)

while True:
# get frame from video
    ret, frame = cap.read()

#Copy code
# detect faces in the frame
    faces = detector(frame)

# loop through each face detected
    for face in faces:
        landmarks = predictor(frame, face)
        left_eye_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye_point = (landmarks.part(45).x, landmarks.part(45).y)
    
    # calculate the distance between the two eye points
        eye_distance = np.sqrt((left_eye_point[0] - right_eye_point[0])**2 + (left_eye_point[1] - right_eye_point[1])**2) / 10

    # if the eye distance is greater than a certain threshold, alert the driver
        print(eye_distance)
        if eye_distance > 20:
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# show the frame
    cv2.imshow("Frame", frame)

# exit if user presses "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
#release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()