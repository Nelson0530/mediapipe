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
cap2 = cv2.VideoCapture(0)
# fps = cap.get(cv2.CAP_PROP_FPS)
t = time.time()

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

pose2 = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

if not cap.isOpened() and cap2.isOpened():
    exit()

sou.play()          #開始播放音頻
score = 0
while True:
    ret, frame = cap.read() #讀取下一個影格
    ret2, img = cap2.read()
    video = cv2.resize(frame, (550, 400))
    video2 = cv2.resize(img, (550, 400))
    video2 = cv2.flip(video2, 1)
    size = video.shape  # 取得攝影機影像尺寸
    w = size[1]  # 取得畫面寬度
    h = size[0]  # 取得畫面高度

    results = pose.process(video)  # 取得姿勢偵測結果
    results2 = pose2.process(video2)
    if results.pose_landmarks and results2.pose_landmarks:
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

            BC = [x4, y4, x6, y6]
            BA = [x4, y4, x2, y2]
            AB = [x2, y2, x4, y4]
            AG = [x2, y2, x8, y8]
            GA = [x8, y8, x2, y2]
            GI = [x8, y8, x10, y10]
            IG = [x10, y10, x8, y8]
            IK = [x10, y10, x12, y12]
            EF = [x3, y3, x5, y5]
            ED = [x3, y3, x1, y1]
            DE = [x1, y1, x3, y3]
            DH = [x1, y1, x7, y7]
            HD = [x7, y7, x1, y1]
            HJ = [x7, y7, x9, y9]
            JH = [x9, y9, x7, y7]
            JL = [x9, y9, x11, y11]

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


        ang1 = angle(BC, BA)
        ang2 = angle(AB, AG)
        ang3 = angle(GA, GI)
        ang4 = angle(IG, IK)
        ang5 = angle(EF, ED)
        ang6 = angle(DE, DH)
        ang7 = angle(HD, HJ)
        ang8 = angle(JH, JL)
        angavg = (ang1 + ang2 + ang3 + ang4 + ang5 + ang6 + ang7 + ang8) / 8

    # if results2.pose_landmarks:
        for pose_landmarks in results2.pose_landmarks.landmark:
            x1 = results2.pose_landmarks.landmark[11].x * w  # 取得手部末端 x 座標
            x2 = results2.pose_landmarks.landmark[12].x * w
            x3 = results2.pose_landmarks.landmark[13].x * w
            x4 = results2.pose_landmarks.landmark[14].x * w
            x5 = results2.pose_landmarks.landmark[15].x * w
            x6 = results2.pose_landmarks.landmark[16].x * w
            x7 = results2.pose_landmarks.landmark[23].x * w
            x8 = results2.pose_landmarks.landmark[24].x * w
            x9 = results2.pose_landmarks.landmark[25].x * w
            x10 = results2.pose_landmarks.landmark[26].x * w
            x11 = results2.pose_landmarks.landmark[27].x * w
            x12 = results2.pose_landmarks.landmark[28].x * w
            y1 = results2.pose_landmarks.landmark[11].y * h  # 取得手部末端 y 座標
            y2 = results2.pose_landmarks.landmark[12].y * h
            y3 = results2.pose_landmarks.landmark[13].y * h
            y4 = results2.pose_landmarks.landmark[14].y * h
            y5 = results2.pose_landmarks.landmark[15].y * h
            y6 = results2.pose_landmarks.landmark[16].y * h
            y7 = results2.pose_landmarks.landmark[23].y * h
            y8 = results2.pose_landmarks.landmark[24].y * h
            y9 = results2.pose_landmarks.landmark[25].y * h
            y10 = results2.pose_landmarks.landmark[26].y * h
            y11 = results2.pose_landmarks.landmark[27].y * h
            y12 = results2.pose_landmarks.landmark[28].y * h

            B_C = [x4, y4, x6, y6]
            B_A = [x4, y4, x2, y2]
            A_B = [x2, y2, x4, y4]
            A_G = [x2, y2, x8, y8]
            G_A = [x8, y8, x2, y2]
            G_I = [x8, y8, x10, y10]
            I_G = [x10, y10, x8, y8]
            I_K = [x10, y10, x12, y12]
            E_F = [x3, y3, x5, y5]
            E_D = [x3, y3, x1, y1]
            D_E = [x1, y1, x3, y3]
            D_H = [x1, y1, x7, y7]
            H_D = [x7, y7, x1, y1]
            H_J = [x7, y7, x9, y9]
            J_H = [x9, y9, x7, y7]
            J_L = [x9, y9, x11, y11]

        def angle(v1, v2):
            dx1 = v1[2] - v1[0]
            dy1 = v1[3] - v1[1]
            dx2 = v2[2] - v2[0]
            dy2 = v2[3] - v2[1]
            angle1 = math.atan2(dy1, dx1)
            angle1 = int(angle1 * 180 / math.pi)
            angle2 = math.atan2(dy2, dx2)
            angle2 = int(angle2 * 180 / math.pi)
            if angle1 * angle2 >= 0:
                included_angle = abs(angle1 - angle2)
            else:
                included_angle = abs(angle1) + abs(angle2)
                if included_angle > 180:
                    included_angle = 360 - included_angle
            return included_angle


        ang_1 = angle(B_C, B_A)
        # print("BC, BA的夾角")
        # print(ang_1)
        ang_2 = angle(A_B, A_G)
        # print("AB, AG的夾角")
        # print(ang_2)
        ang_3 = angle(G_A, G_I)
        # print("GA, GI的夾角")
        # print(ang_3)
        ang_4 = angle(I_G, I_K)
        # print("IG, IK的夾角")
        # print(ang_4)
        ang_5 = angle(E_F, E_D)
        # print("EF, ED的夾角")
        # print(ang_5)
        ang_6 = angle(D_E, D_H)
        # print("DE, DH的夾角")
        # print(ang_6)
        ang_7 = angle(H_D, H_J)
        ang_8 = angle(J_H, J_L)
        ang_avg = (ang_1 + ang_2 + ang_3 + ang_4 + ang_5 + ang_6 + ang_7 + ang_8) / 8


        # n = abs(ang_avg - angavg)  # 將兩個平均角度相減取絕對值
        i = int(abs(ang_avg - angavg))
        if i <= 5:
            score += 5
        elif i > 5 and i <= 15:
            score += 3
        elif i > 15 and i <= 25:
            score += 1
        else:
            score += 0

        print(score)

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
        cv2.imshow('test', video)      # 顯示影格
        del(ret)
        del(video)
        cv2.imshow('test_xxx', video2)  # 顯示影格
    else:
        break

    if cap.get(1) / cap.get(5) > time.time()-t:
        time.sleep(0.04)
    if cap.get(1) / cap.get(5) < time.time()-t-0.04:
        cap.set(1, cap.get(1) + 3)

    if cv2.waitKey(5) == 27:
        cv2.destroyAllWindows()
        break

while True:
    pygame.mixer.music.stop()
    break

# print(score)
cap.release()
cap2.release()
cv2.destroyAllWindows()
