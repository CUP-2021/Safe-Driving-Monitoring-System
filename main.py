import sys
import cv2
from PySide2.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QGridLayout,
    QSplitter,
)
import simpleaudio as sa
import os
import config
from PySide2.QtCore import QTimer, Qt
from PySide2.QtGui import QImage, QPixmap
import myframe
from ultralytics import YOLO
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)

# 全局变量初始化
COUNTER = 0  # 眨眼帧计数器
TOTAL = 0  # 眨眼总数
mCOUNTER = 0  # 打哈欠帧计数器
mTOTAL = 0  # 打哈欠总数
ActionCOUNTER = 0  # 分心行为计数器
Roll = 0  # 整个循环内的帧计数
Rolleye = 0  # 循环内闭眼帧数
Rollmouth = 0  # 循环内打哈欠数

class VideoPlayerWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("疲劳检测系统")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()
        QTimer.singleShot(1000, self.setupVideo)
        self.alert_sound = sa.WaveObject.from_wave_file(config.ALERT_SOUND_FILE)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("face/trainer/trainer.yml")
        self.names = []

        self.load_names()
        self.face_recognition_enabled = True

    def initUI(self):
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        self.video_label = QLabel()
        video_layout.addWidget(self.video_label)
        layout.addWidget(video_widget)

        # 创建视频显示部件，并为其设置红色边框样式
        self.video_widget = QWidget()
        self.video_widget.setStyleSheet(
            "border: 5px solid transparent;"
        )  # 初始设置不显示边框颜色
        video_layout = QVBoxLayout(self.video_widget)
        self.video_label = QLabel()
        video_layout.addWidget(self.video_label)
        layout.addWidget(self.video_widget)

        splitter = QSplitter(Qt.Vertical)
        status_widget, output_widget = self.createStatusAndOutputWidgets()
        splitter.addWidget(status_widget)
        splitter.addWidget(output_widget)
        layout.addWidget(splitter)

    def createStatusAndOutputWidgets(self):
        status_widget = QWidget()
        status_layout = QVBoxLayout(status_widget)
        self.status_labels = {}
        modules = ["手机", "抽烟", "喝水", "状态", "眨眼", "哈欠"]
        self.initial_statuses = ["未使用", "未抽烟", "未喝水", "清醒", 0, 0]
        grid_layout = QGridLayout()
        for i, module in enumerate(modules):
            label = QLabel(f"{module}: {self.initial_statuses[i]}")
            self.status_labels[module] = label
            grid_layout.addWidget(label, i // 2, i % 2)
        status_layout.addLayout(grid_layout)

        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        self.output_textedit = QTextEdit()
        self.output_textedit.setReadOnly(True)
        output_layout.addWidget(self.output_textedit)

        return status_widget, output_widget

    def setupVideo(self):
        # 加载模型，设置视频源和处理视频帧的定时器
        model_path = "pt/633 and 0.98best.engine"  # 模型路径
        self.model = YOLO(model_path, task="detect")
        self.output_textedit.append("正在加载摄像头...")

        # 设置视频源
        pipeline_str = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, framerate=(fraction)30/1, format=(string)NV12 ! "
            "tee name=t "
            "t. ! queue leaky=downstream max-size-buffers=1 ! nvvidconv flip-method=0 ! "
            "video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink emit-signals=True sync=false max-buffers=1 drop=True "
            "t. ! queue leaky=downstream ! nvv4l2h264enc maxperf-enable=1 bitrate=4000000 control-rate=1 preset-level=1 ! "
            "h264parse ! flvmux streamable=true ! rtmpsink location="
            + config.RTMP_URL
            + " live=1 sync=false"
        )

        self.cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.output_textedit.append("无法打开摄像头，请检查。")
            return

        # 设置定时器处理视频帧
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_video)
        self.timer.start(30)  # 根据需要调整间隔

    def play_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.output_textedit.append("视频帧读取失败。")
            self.timer.stop()  # 停止定时器
            return

        # 使用已加载的模型处理帧
        ret, processed_frame = myframe.frametest(frame, self.model)
        lab, eye, mouth = ret

        # 更新状态标签
        self.updateStatus(lab, eye, mouth)

        # 在帧上进行人脸检测和识别
        processed_frame = self.face_detect_recognize(processed_frame)

        # 显示处理后的帧
        show_image = QImage(
            processed_frame.data,
            processed_frame.shape[1],
            processed_frame.shape[0],
            QImage.Format_RGB888,
        ).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(show_image))

    def trigger_red_frame(self, show):

        if show:
            self.video_widget.setStyleSheet("border: 5px solid red;")
            self.alert_sound.play()
            QTimer.singleShot(500, self.hide_red_frame)  # 500毫秒后隐藏红框

        else:
            self.video_widget.setStyleSheet("")  # 清除边框样式以隐藏红框

    def hide_red_frame(self):
        self.video_widget.setStyleSheet("")  # 清除边框样式以隐藏红框
        QTimer.singleShot(500, self.show_red_frame)  # 500毫秒后再次显示红框

    def show_red_frame(self):
        if self.red_frame_is_active:
            self.video_widget.setStyleSheet("border: 5px solid red;")
            QTimer.singleShot(500, self.hide_red_frame)

    def updateStatus(self, labels, eye_ar, mouth_ar):
        # 根据识别到的行为更新状态
        self.status_labels["手机"].setText(
            f"手机: {'使用中' if 'phone' in labels else '未使用'}"
        )
        self.status_labels["抽烟"].setText(
            f"抽烟: {'抽烟中' if 'smoke' in labels else '未抽烟'}"
        )
        self.status_labels["喝水"].setText(
            f"喝水: {'喝水中' if 'drink' in labels else '未喝水'}"
        )

        distraction_detected = "phone" in labels or "smoke" in labels or "drink" in labels
        if distraction_detected:
            self.red_frame_is_active = True
            self.trigger_red_frame(True)
        else:
            self.red_frame_is_active = False
            self.trigger_red_frame(False)

        # 更新眨眼和哈欠次数
        global TOTAL, mTOTAL  # 需要访问全局变量
        if eye_ar < config.EYE_AR_THRESH:
            global COUNTER
            COUNTER += 1
        else:
            if COUNTER >= config.EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                COUNTER = 0
        if mouth_ar > config.MAR_THRESH:
            global mCOUNTER
            mCOUNTER += 1
        else:
            if mCOUNTER >= config.MOUTH_AR_CONSEC_FRAMES:
                mTOTAL += 1
                mCOUNTER = 0

        # 更新眨眼和哈欠的显示状态
        self.status_labels["眨眼"].setText(f"眨眼: {TOTAL}")
        self.status_labels["哈欠"].setText(f"哈欠: {mTOTAL}")

        # 根据疲劳判断更新状态
        global Roll, Rolleye, Rollmouth
        Roll += 1
        if eye_ar < config.EYE_AR_THRESH:
            Rolleye += 1
        if mouth_ar > config.MAR_THRESH:
            Rollmouth += 1
        if Roll >= config.FATIGUE_CALCULATION_FRAMES:
            perclos = (Rolleye / Roll) + (Rollmouth / Roll) * 0.2
            self.status_labels["状态"].setText(
                f"状态: {'疲劳' if perclos > config.PERCLOS_THRESHOLD else '清醒'}"
            )
            self.output_textedit.append(
                "过去150帧中，Perclos 得分为" + str(round(perclos, 3))
            )
            if perclos > config.PERCLOS_THRESHOLD:
                self.red_frame_is_active = True
                self.trigger_red_frame(True)
            else:
                self.red_frame_is_active = False
                self.trigger_red_frame(False)

            # 重置计数器
            Roll, Rolleye, Rollmouth = 0, 0, 0
            TOTAL, mTOTAL = 0, 0

    def load_names(self):
        path = "face/jm"
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            # 假设文件名格式为 '1.xxx.jpg'
            filename = os.path.basename(imagePath)  # 获取文件名，例如 '1.xxx.jpg'
            name_parts = filename.split(".")  # 分割为 ['1', 'xxx', 'jpg']
            if len(name_parts) > 1:
                name = name_parts[1]  # 获取中间的 'xxx' 部分
                self.names.append(name)

    def face_detect_recognize(self, img):
        if not self.face_recognition_enabled:
            return img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier(config.CASCADE_CLASSIFIER_PATH)
        faces = face_detector.detectMultiScale(gray)

        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
            roi_gray = gray[y : y + h, x : x + w]
            id_, confidence = self.recognizer.predict(roi_gray)
            print(f"Recognized ID: {id_}, Confidence: {confidence}")
            if confidence > config.RECOGNITION_CONFIDENCE_THRESHOLD:
                name = "Unknown"
            else:
                name = self.names[id_ - 1]
                self.face_recognition_enabled = False  # 识别到人脸后停止识别

            if name == "Unknown":
                cv2.putText(
                    img,
                    name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0) if name != "Unknown" else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # 如果名称已识别且未被欢迎，则在 GUI 中显示欢迎信息
            if name != "Unknown":
                self.output_textedit.append(f"欢迎您, {name}!")

        return img


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoPlayerWindow()
    window.show()
    sys.exit(app.exec_())
