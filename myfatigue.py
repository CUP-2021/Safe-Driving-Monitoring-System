from scipy.spatial import distance as dist
import imutils
from imutils import face_utils
import numpy as np
import dlib
import cv2
import config


# 计算眼睛纵横比
def eye_aspect_ratio(eye):
    """
    计算眼睛的纵横比(EAR)，用于评估眼睛的开合程度。
    参数:
    eye (list): 眼睛的坐标点列表。
    返回:
    float: 眼睛的纵横比。
    """
    # 眼睛的垂直距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 眼睛的水平距离
    C = dist.euclidean(eye[0], eye[3])
    # 计算纵横比
    ear = (A + B) / (2.0 * C)
    return ear


# 计算嘴巴纵横比
def mouth_aspect_ratio(mouth):
    """
    计算嘴巴的纵横比(MAR)，用于评估嘴巴的开合程度。
    参数:
    mouth (list): 嘴巴的坐标点列表。
    返回:
    float: 嘴巴的纵横比。
    """
    # 嘴巴的垂直距离
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    # 嘴巴的水平距离
    C = np.linalg.norm(mouth[0] - mouth[6])
    # 计算纵横比
    mar = (A + B) / (2.0 * C)
    return mar


# 加载面部标志检测器
print("[INFO] 加载面部标志预测器...")
detector = dlib.get_frontal_face_detector()  # 脸部位置检测器
predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR_PATH)

# 定义左右眼和嘴巴的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


# 检测疲劳的函数
def detfatigue(frame):
    """
    对给定的帧进行疲劳检测，包括眼睛和嘴巴的检测。
    参数:
    frame (ndarray): 视频帧。
    返回:
    tuple: 包含处理后的帧、眼睛纵横比和嘴巴纵横比的元组。
    """
    # 缩放帧并转换为灰度图
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检测脸部
    rects = detector(gray, 0)

    eyear = 0.0  # 眼睛纵横比初始值
    mouthar = 0.0  # 嘴巴纵横比初始值

    # 遍历检测到的脸部
    for rect in rects:
        shape = predictor(gray, rect)  # 获取脸部特征点
        shape = face_utils.shape_to_np(shape)  # 转换为numpy数组

        # 提取眼睛和嘴巴的坐标
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # 计算眼睛和嘴巴的纵横比
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        eyear = (leftEAR + rightEAR) / 2.0  # 平均眼睛纵横比
        mouthar = mouth_aspect_ratio(mouth)  # 嘴巴纵横比

        # 绘制眼睛和嘴巴的轮廓
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

    # 返回处理后的帧和计算的纵横比

    return frame, eyear, mouthar
