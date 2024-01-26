import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture(0)
detector = pm.poseDetector()

pTime = 0
cTime = 0

count = 0
dir = 0
while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        # Left arm
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (30, 130), (0, 100))
        # print(angle, per)

        # Check for the dumbbell curls
        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
        elif per == 0:
            if dir == 1:
                count += 0.5
                dir = 0
        # print(count)
        
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 5)

    # show fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (25, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('0'):
        break

cap.release()
cv2.destroyAllWindows()