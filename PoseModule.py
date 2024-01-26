import cv2
import mediapipe as mp
import time
import math
import numpy as np

class poseDetector():

    def __init__(self, mode=False, model=1, smooth=True, enable_seg=False, smooth_seg=True,
                detectCon=0.5, trackCon=0.5):
        
        self.mode = mode
        self.model = model
        self.smooth = smooth
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model, self.smooth,
            self.enable_seg, self.smooth_seg, self.detectCon, self.trackCon
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle
        
        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), 2)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 0), 2)
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (255, 0, 0), 2)
            cv2.putText(img, str(int(angle)), (x2-50, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        return angle

def main():
    # cap = cv2.VideoCapture('PoseVideo/Video1.mp4') # use video
    cap = cv2.VideoCapture(0) # use webcam

    detector = poseDetector()
    pTime = 0 # previous time
    cTime = 0 # current time
    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        # print(lmList[14])
        # cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        if len(lmList) != 0:
            # Right arm
            # detector.findAngle(img, 12, 14, 16)

            # Left arm
            angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, (210, 310), (0, 100))
            print(angle, per)
            

        # fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        # img = cv2.resize(img, (1920, 1080))
        cv2.imshow('Image', img)
        if cv2.waitKey(1) == ord('0'):
            break

if __name__ == '__main__':
    main()