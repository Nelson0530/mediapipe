import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("./mp4/bonbon1.mp4")
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))        # 取得畫面尺寸
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# print(w, h)
# 建立 VideoWriter 物件，輸出影片至 output.avi，FPS 值為 20.0
# out = cv2.VideoWriter('./cars.mp4', fourcc, fps, (520, 400))

# 啟用姿勢偵測
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # start = time.time()
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        # img = cv2.resize(img, (520, 400))               # 縮小尺寸，加快演算速度
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB


        results = pose.process(img2)                  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            img2,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # end = time.time()

        # out.write(img2)

        cv2.imshow('oxxostudio', img2)
        # if cv2.waitKey(5) == ord('q'):
        if cv2.waitKey(5) == 27:
            break     # 按下 q 鍵停止
# print(f"執行時間：{end - start} 秒")
cap.release()
# out.release()
cv2.destroyAllWindows()
