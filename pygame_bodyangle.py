import cv2
import pygame
import time
import mediapipe as mp
import math

# 注意要設定相關參數，不然轉出來的影片會沒有聲音
pygame.init()
mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

cap = cv2.VideoCapture('./911_song.mp4')        #開啟影片檔案
sou = pygame.mixer.Sound('./911_song.mp3')
sou.play()          #開始播放音頻
cap2 = cv2.VideoCapture(0)
# fps = cap.get(cv2.CAP_PROP_FPS)
t = time.time()

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    if not cap.isOpened() and cap2.isOpened():
        exit()

    while True:
        ret, frame = cap.read() #讀取下一個影格
        ret2, img = cap2.read()
        video = cv2.resize(frame, (550, 400))
        video2 = cv2.resize(img, (550, 400))
        size = frame.shape  # 取得攝影機影像尺寸
        w = size[1]  # 取得畫面寬度
        h = size[0]  # 取得畫面高度

        results = pose.process(video)  # 取得姿勢偵測結果
        results2 = pose.process(video2)
        if results.pose_landmarks:
            for pose_landmarks in results.pose_landmarks.landmark:
                x1 = results.pose_landmarks.landmark[11].x * w  # 取得手部末端 x 座標
                x2 = results.pose_landmarks.landmark[12].x * w
                x3 = results.pose_landmarks.landmark[13].x * w
                x4 = results.pose_landmarks.landmark[14].x * w
                x5 = results.pose_landmarks.landmark[15].x * w
                x6 = results.pose_landmarks.landmark[16].x * w
                x7 = results.pose_landmarks.landmark[23].x * w
                x8 = results.pose_landmarks.landmark[24].x * w
                x9 = results.pose_landmarks.landmark[25].x * w
                x10 = results.pose_landmarks.landmark[26].x * w
                x11 = results.pose_landmarks.landmark[27].x * w
                x12 = results.pose_landmarks.landmark[28].x * w
                y1 = results.pose_landmarks.landmark[11].y * h  # 取得手部末端 y 座標
                y2 = results.pose_landmarks.landmark[12].y * h
                y3 = results.pose_landmarks.landmark[13].y * h
                y4 = results.pose_landmarks.landmark[14].y * h
                y5 = results.pose_landmarks.landmark[15].y * h
                y6 = results.pose_landmarks.landmark[16].y * h
                y7 = results.pose_landmarks.landmark[23].y * h
                y8 = results.pose_landmarks.landmark[24].y * h
                y9 = results.pose_landmarks.landmark[25].y * h
                y10 = results.pose_landmarks.landmark[26].y * h
                y11 = results.pose_landmarks.landmark[27].y * h
                y12 = results.pose_landmarks.landmark[28].y * h

            CB = [x5, y5, x3, y3]
            BA = [x3, y3, x1, y1]
            AG = [x1, y1, x7, y7]
            GI = [x7, y7, x9, y9]
            IK = [x9, y9, x11, y11]
            FE = [x6, y6, x4, y4]
            ED = [x4, y4, x2, y2]
            DH = [x2, y2, x8, y8]
            HJ = [x8, y8, x10, y10]
            JL = [x10, y10, x12, y12]

            def angle(v1, v2):
                dx1 = v1[2] - v1[0]
                dy1 = v1[3] - v1[1]
                dx2 = v2[2] - v2[0]
                dy2 = v2[3] - v2[1]
                angle1 = math.atan2(dy1, dx1)
                angle1 = int(angle1 * 180 / math.pi)
                # print(angle1)
                angle2 = math.atan2(dy2, dx2)
                angle2 = int(angle2 * 180 / math.pi)
                # print(angle2)
                if angle1 * angle2 >= 0:
                    included_angle = abs(angle1 - angle2)
                else:
                    included_angle = abs(angle1) + abs(angle2)
                    if included_angle > 180:
                        included_angle = 360 - included_angle
                return included_angle

            ang1 = angle(CB, BA)
            print("CB, BA的夾角")
            print(ang1)
            ang2 = angle(BA, AG)
            print("BA, AG的夾角")
            print(ang2)
            ang3 = angle(AG, GI)
            print("AG, GI的夾角")
            print(ang3)
            ang4 = angle(GI, IK)
            print("GI, IK的夾角")
            print(ang4)
            ang5 = angle(FE, ED)
            print("FE, ED的夾角")
            print(ang5)
            ang6 = angle(ED, DH)
            print("ED, DH的夾角")
            print(ang6)
            ang7 = angle(DH, HJ)
            ang8 = angle(HJ, JL)
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            video,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        mp_drawing.draw_landmarks(
            video2,
            results2.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if ret and ret2:
            cv2.imshow('test', video) #顯示影格
            cv2.imshow('testxxx', video2)  # 顯示影格
        else:
            break

        if cap.get(1) / cap.get(5) > time.time()-t:
            time.sleep(0.04)
        if cap.get(1) / cap.get(5) < time.time()-t-0.04:
            cap.set(1, cap.get(1) + 1)

        if cv2.waitKey(5) == 27:
            cv2.destroyAllWindows()
            break

    while True:
        pygame.mixer.music.stop()
        break

cap.release()
cap2.release()
cv2.destroyAllWindows()
