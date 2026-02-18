import cv2
import numpy as np
import mediapipe as mp
import time
import pyautogui
import math
import winsound
import pygetwindow as gw
import threading

# Configuration
SMOOTHING_FACTOR = 0.2  # EMA factor: Lower = Smoother/Slower, Higher = Faster/Jittery
SNIPER_SMOOTHING = 0.02 # Very slow for precision
CLICK_THRESHOLD = 30
FRAME_REDUCTION = 100 # Box size padding from camera edge

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.7):
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
        self.allHands = []

        if self.results.multi_hand_landmarks:
            for idx, handLms in enumerate(self.results.multi_hand_landmarks):
                myHand = {}
                lmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    xList.append(cx)
                    yList.append(cy)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax
                myHand["lmList"] = lmList
                myHand["bbox"] = bbox
                myHand["center"] = (xmin + xmax) // 2, (ymin + ymax) // 2
                
                # Retrieve label
                if self.results.multi_handedness:
                    label = self.results.multi_handedness[idx].classification[0].label
                    myHand["type"] = label 

                self.allHands.append(myHand)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (255, 0, 255), 2)
        return self.allHands, img

    def fingersUp(self, myHand):
        fingers = []
        lmList = myHand["lmList"]
        handType = myHand.get("type", "Right")
        
        # Thumb: Check x-coordinates
        # Assuming Right Hand
        if handType == "Right":
            if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]: fingers.append(1)
            else: fingers.append(0)
        else:
            if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0] - 1][1]: fingers.append(1)
            else: fingers.append(0)

        for id in range(1, 5):
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]: fingers.append(1)
            else: fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None, lmList1=None, lmList2=None):
        if lmList1 is None: return 0, img, []
        x1, y1 = lmList1[p1][1:]
        if lmList2 is None: x2, y2 = lmList1[p2][1:]
        else: x2, y2 = lmList2[p2][1:]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        return length, info

def beep_async(freq, duration):
    try:
        threading.Thread(target=winsound.Beep, args=(freq, duration), daemon=True).start()
    except: pass

def main():
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = HandDetector(maxHands=2, detectionCon=0.8, trackCon=0.8)
    wScr, hScr = pyautogui.size()
    
    # State
    plocX, plocY = 0, 0 # Previous Location
    dragging = False
    pTime = 0
    
    # Click Stabilization
    click_frames = 0 
    
    print("Enhanced Accuracy System Active")

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)

        hands, img = detector.findHands(img, draw=True)
        cv2.rectangle(img, (FRAME_REDUCTION, FRAME_REDUCTION), 
                      (wCam - FRAME_REDUCTION, hCam - FRAME_REDUCTION), (255, 0, 255), 2)
        
        if len(hands) == 1:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = detector.fingersUp(hand)
            x1, y1 = lmList[8][1:] # Index Tip
            
            # Distance (Thumb, Index)
            distPinch, _ = detector.findDistance(4, 8, img, lmList1=lmList)
            isPinching = distPinch < CLICK_THRESHOLD

            # --- NAVIGATION & SNIPER MODE ---
            # Index Up Only OR Index+Middle Up (Sniper)
            if fingers[1] == 1:
                isSniper = (fingers[2] == 1)
                
                # Active Smoothing Factor
                cur_alpha = SNIPER_SMOOTHING if isSniper else SMOOTHING_FACTOR
                if isSniper: 
                     cv2.putText(img, "Sniper Mode", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                # Convert Coordinates
                # Mapping active box (Frame Reduction) to Screen Resolution
                x3 = np.interp(x1, (FRAME_REDUCTION, wCam - FRAME_REDUCTION), (0, wScr))
                y3 = np.interp(y1, (FRAME_REDUCTION, hCam - FRAME_REDUCTION), (0, hScr))
                
                # Exponential Moving Average for Smoothness
                # curr = alpha * target + (1-alpha) * prev
                clocX = cur_alpha * x3 + (1 - cur_alpha) * plocX
                clocY = cur_alpha * y3 + (1 - cur_alpha) * plocY
                
                # Click Stabilization
                # If we JUST started clicking/pinching, freeze movement for a few frames
                if isPinching and not dragging:
                    click_frames += 1
                else:
                    click_frames = 0
                
                # Only move if not in "Click Stabilization" window (first 3 frames of a pinch)
                # This prevents the cursor from jumping when you tap your fingers.
                if click_frames < 3: 
                    pyautogui.moveTo(clocX, clocY)
                
                plocX, plocY = clocX, clocY

                # --- CLICK / DRAG ---
                if isPinching:
                    cv2.circle(img, (lmList[8][1], lmList[8][2]), 15, (0, 255, 0), cv2.FILLED)
                    if not dragging:
                        if click_frames > 2: # Wait for stabilization
                            pyautogui.mouseDown()
                            beep_async(1000, 50)
                            dragging = True
                else:
                    if dragging:
                        pyautogui.mouseUp()
                        beep_async(600, 50)
                        dragging = False

            # --- CONTEXT AWARE (Pinky Up context) ---
            if fingers[4] == 1 and fingers[1] == 0:
                cv2.putText(img, "Context Mode", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                 # Add simple vertical check for scroll
                _, cy = hand["center"]
                if cy < hCam // 2 - 50: pyautogui.scroll(50)
                elif cy > hCam // 2 + 50: pyautogui.scroll(-50)

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
