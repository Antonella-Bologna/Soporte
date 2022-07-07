import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
bandera = False

cap = cv2.VideoCapture(0)
pTime = 0

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    #image = cv2.resize(image, (800, 800))
    success2, image2 = cap.read()
    #image2 = cv2.resize(image2, (800, 800))

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    area= image[100:300, 400:600]
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w,c = image.shape
            #print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(image, (cx, cy), 5, (255,0,0), cv2.FILLED)

    

    #cv2.putText(image, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    if bandera:
        gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        cv2.circle(image2,(20,20),10,(0,0,139),-1)
        cv2.imshow("Image", image2)
    else:
        cv2.imshow("Image", image2)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height,width,_=image.shape
    if results.multi_hand_landmarks:
        print(results.multi_hand_landmarks)
        #bandera = not bandera
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    
    area.flags.writeable = False
    area = cv2.cvtColor(area, cv2.COLOR_BGR2RGB)
    results = hands.process(area)
    area.flags.writeable = True
    area = cv2.cvtColor(area, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        print(results.multi_hand_landmarks)
        bandera = not bandera
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                area,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
    # Flip the image horizontally for a selfie-view display.
    cv2.rectangle(image, (400, 100), (600,300), (0,0,139), 2)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()