import cv2
import numpy as np
import mediapipe as mp
import time
import pyautogui
import math

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon, 
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        # specific for right hand mostly, might need adjustment for left
        if len(self.lmList) != 0:
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    # Define IDs for fingers
    # 4: Thumb, 8: Index, 12: Middle, 16: Ring, 20: Pinky

    # Needed Variables
    wCam, hCam = 640, 480
    frameR = 100 # Frame Reduction
    smoothening = 7
    
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    
    detector = HandDetector(maxHands=1)
    wScr, hScr = pyautogui.size()
    
    dragging = False

    while True:
        # 1. Find hand Landmarks
        success, img = cap.read()
        if not success:
            break
        
        # Flip the image horizontally for a mirror effect & easier intuition
        img = cv2.flip(img, 1)

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        
        # 2. Get the tip of the index and middle fingers
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]   # Index Finger
            x2, y2 = lmList[12][1:]  # Middle Finger
            
            # 3. Check which fingers are up
            fingers = detector.fingersUp()
            
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                          (255, 0, 255), 2)
            
            # Check for Pinch (Index and Thumb) for Dragging
            # Thumb tip (4) and Index tip (8)
            lengthDrag, img, lineInfoDrag = detector.findDistance(4, 8, img)
            isPinching = lengthDrag < 30 # Threshold for pinch

            # 4. Only Index Finger : Moving Mode (with optional Drag)
            # We check if Middle finger is down to ensure we are not in "Click" mode
            if fingers[1] == 1 and fingers[2] == 0:
                
                # 5. Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                
                # 6. Smoothen Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                
                # 7. Move Mouse
                # Since we flipped the image, moving hand Right moves cursor Right
                pyautogui.moveTo(clocX, clocY)
                plocX, plocY = clocX, clocY
                
                # 8. Drag Logic
                if isPinching:
                    cv2.circle(img, (lineInfoDrag[4], lineInfoDrag[5]), 15, (0, 255, 0), cv2.FILLED)
                    if not dragging:
                        pyautogui.mouseDown()
                        dragging = True
                else:
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False

            # 9. Both Index and Middle fingers are up : Clicking Mode
            # This mode stops movement to allow stable clicking
            if fingers[1] == 1 and fingers[2] == 1:
                # Find distance between Index and Middle
                lengthClick, img, lineInfoClick = detector.findDistance(8, 12, img)
                
                # 10. Click mouse if distance short
                if lengthClick < 40:
                    cv2.circle(img, (lineInfoClick[4], lineInfoClick[5]),
                               15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()
                    # Add a small delay/debounce if needed, or rely on finger separation
                    time.sleep(0.1) 
        
        # 11. Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        
        # 12. Display
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
