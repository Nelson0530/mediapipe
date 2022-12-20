import cv2
import mediapipe as mp
import random
import time

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測
# mp_hands = mp.solutions.hands                    # mediapipe 偵測手掌方法

cap = cv2.VideoCapture(0)

# 啟用姿勢偵測
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

# #  mediapipe 啟用偵測手掌
# with mp_hands.Hands(
#     model_complexity=0,
#     # max_num_hands=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    run = run1 = True
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img = cv2.resize(img, (850, 550))              # 縮小尺寸，能加快演算速度
        img = cv2.flip(img, 1)
        size = img.shape  # 取得攝影機影像尺寸
        w = size[1]  # 取得畫面寬度
        h = size[0]  # 取得畫面高度
        if run:
            run = False  # 如果沒有碰到，就一直是 False ( 不會更換位置 )
            rx = random.randint(50, w - 50)  # 隨機 x 座標
            ry = random.randint(50, h - 100)  # 隨機 y 座標
            # print(rx, ry)
        if run1:
            run1 = False  # 如果沒有碰到，就一直是 False ( 不會更換位置 )
            rx1 = random.randint(50, w - 50)  # 隨機 x 座標
            ry1 = random.randint(50, h - 100)  # 隨機 y 座標

        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = pose.process(img2)                  # 取得姿勢偵測結果
        if results.pose_landmarks:
            for pose_landmarks in results.pose_landmarks.landmark:
                x = results.pose_landmarks.landmark[19].x * w  # 取得手部末端 x 座標
                x1 = results.pose_landmarks.landmark[20].x * w
                y = results.pose_landmarks.landmark[19].y * h  # 取得手部末端 y 座標
                y1 = results.pose_landmarks.landmark[20].y * h
                # print(x, y)
                if x > rx and x < (rx + 80) and y > ry and y < (ry + 80) :
                    run = True
                if x1 > rx and x1 < (rx + 80) and y1 > ry and y1 < (ry + 80):
                    run = True
                if x1 > rx1 and x1 < (rx1 + 80) and y1 > ry1 and y1 < (ry1 + 80):
                    run1 = True
                if x > rx1 and x < (rx1 + 80) and y > ry1 and y < (ry1 + 80):
                    run1 = True
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.rectangle(img, (rx, ry), (rx + 80, ry + 80), (0, 0, 255), 5)  # 畫出觸碰區
        cv2.rectangle(img, (rx1, ry1), (rx1 + 80, ry1 + 80), (0, 255, 0), 5)  # 畫出觸碰區
        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(5) == ord('q'):
            break     # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()
