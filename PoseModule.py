import cv2
import mediapipe as mp
import time

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
        lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


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

        img = detector.findPose(img)
        lmList = detector.findPosition(img)

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