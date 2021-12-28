import time
import cv2
import mediapipe as mp
import sounddevice as sd
import numpy as np


class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode, 
            max_num_hands=self.maxHands, 
            min_detection_confidence=self.detectionCon, 
            min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img



    def findPosition(self, img, handNo = 0, draw = True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape

            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist

    def determine_pitch_hands(self, img):
        """
        Takes the results and determines the hand that is lower and higher.
        2nd index is the pitch hand position normalized.
        First Index is the volume hand position normalized

        """

        hands = self.results.multi_hand_landmarks
        if hands:
            print(len(hands))
            if len(hands) != 2:
                return 0, 0
            
            h, w, *_ = img.shape

            l = []
            for hand in hands:
                for _, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    break
                l.append([cx, cy])
            
            (x0, y0), (x1, y1) = l
            print(l)

            if x0 < x1: 
                # The 0th hand is the volume control because it's further left

                return self.compute_vol(x0, y0, h), self.compute_pitch(x1, y1, h)
            else:
                # The 1th hand is the volume control
                return self.compute_vol(x1, y1, h), self.compute_pitch(x0, y0, h)

        return 0, 0


    def compute_vol(self, vol_x, vol_y, high):
        # volume gets louder as you move closer to the bottom
        return 1.0 - (vol_y / high)

    def compute_pitch(self, pitch_x, pitch_y, high):
        return pitch_y / high

    



def main():
    def snd_callback(outdata: np.ndarray, frames: int, time, status):
        if status:
            print(status, file=sys.stderr)
        nonlocal start_idx
        t = (start_idx + np.arange(frames)) / samplerate
        t = t.reshape(-1, 1)
        outdata[:] = volume * np.sin(2 * np.pi * (pitch * 2000 + 200) * t)
        start_idx += frames


    start_idx = 0
    pitch, volume = 0, 0

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    samplerate=44100
    with sd.OutputStream(device=sd.default.device, channels=1, callback=snd_callback, samplerate=samplerate, blocksize=3000):
        pTime = 0
        while True:
            success, img = cap.read()
            
            # Flip it so that it looks like I'm looking in a mirror
            img = cv2.flip(img, 1)
            img = detector.findHands(img)
            pitch, volume = detector.determine_pitch_hands(img)
            print(pitch, volume)

            #lmlist = detector.findPosition(img)
            #if len(lmlist) != 0:
            #    print(f'{lmlist[0]=}')
            #    #print(lmlist[4])

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime


            cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('input', img)

            c = cv2.waitKey(1)
            if c == 27:
                break


if __name__ == '__main__':
    main()