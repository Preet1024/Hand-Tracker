import cv2
from cvzone.HandTrackingModule import HandDetector

class HandTracker:
    def __init__(self, detectionCon=0.5, maxHands=2):
        self.detector = HandDetector(detectionCon=detectionCon, maxHands=maxHands)

    def track(self, image):
        hands, image = self.detector.findHands(image)

        if hands:
            for i, hand in enumerate(hands):
                lmList = hand["lmList"]
                bbox = hand["bbox"]
                handType = hand["type"]
                fingers = self.detector.fingersUp(hand)
                height1, _, _ = self.detector.findDistance(lmList[5][0:2], lmList[6][0:2], image)
                height2, _, _ = self.detector.findDistance(lmList[6][0:2], lmList[7][0:2], image)
                height3, _, _ = self.detector.findDistance(lmList[7][0:2], lmList[8][0:2], image)
                height = height1 + height2 + height3
                width, _, _ = self.detector.findDistance(lmList[7][0:2], lmList[8][0:2], image)
                dist = tuple(round(num, 1) for num in (height, width))
                color = (128, 128, 128) if i == 0 else (0, 0, 0)
                cv2.putText(image, f'Index Finger{i+1}: {handType} - {dist}mm - {fingers}', (10, 30+i*30), cv2.FONT_HERSHEY_PLAIN, 1.12, color, 2)

        return image