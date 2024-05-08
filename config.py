# config.py

# 阈值设置
EYE_AR_THRESH = 0.15  # 眼睛长宽比阈值，低于此值视为闭眼
EYE_AR_CONSEC_FRAMES = 2  # 连续多少帧闭眼才算一次眨眼
MAR_THRESH = 0.65  # 嘴巴长宽比阈值，高于此值视为打哈欠
MOUTH_AR_CONSEC_FRAMES = 3  # 连续多少帧打哈欠才算一次哈欠

# 计算疲劳的参数
FATIGUE_CALCULATION_FRAMES = 150  # 每150帧计算一次疲劳度
PERCLOS_THRESHOLD = 0.15  # 疲劳度阈值

# 模型路径
MODEL_PATH = "pt/633 and 0.98best.engine"

# RTMP URL
RTMP_URL = ""

# 视频帧处理间隔
VIDEO_FRAME_INTERVAL = 30  # 根据需要调整间隔

# 声音文件路径
ALERT_SOUND_FILE = "sound/alert.wav"

# 人脸识别模型路径
RECOGNIZER_MODEL_PATH = "face/trainer/trainer.yml"

# 人脸识别配置
FACE_RECOGNITION_ENABLED = True

# 哈尔级联分类器路径
CASCADE_CLASSIFIER_PATH = "face/haarcascade_frontalface_alt2.xml"

# 识别置信度阈值
RECOGNITION_CONFIDENCE_THRESHOLD = 80

# 图片路径
FACES_IMAGE_PATH = "face/jm"

# 人脸特征点预测器路径
SHAPE_PREDICTOR_PATH = "face/shape_predictor_68_face_landmarks.dat"
