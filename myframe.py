import cv2
import myfatigue
import time
import numpy as np

# 定义物体名称
name = {0: "face", 1: "smoke", 2: "phone", 3: "drink"}


# 帧测试函数
def frametest(frame, model):
    """
    对给定的帧执行疲劳检测和物体检测。
    参数:
    frame (ndarray): 视频帧。
    model: 物体检测模型。
    返回:
    tuple: 包含标签列表、眼睛和嘴巴纵横比的元组和处理后的帧。
    """
    ret = []  # 初始化返回列表
    labellist = []  # 初始化标签列表
    tstart = time.time()  # 开始时间

    # 执行疲劳检测
    frame, eye, mouth = myfatigue.detfatigue(frame)

    # 执行物体检测
    results = model.predict(frame, conf=0.5, device=0)

    # 处理检测结果
    for result in results:
        if len(result.boxes.xyxy) > 0:
            boxes_conf = np.array(result.boxes.conf.tolist())
            boxes_conf_n = len(boxes_conf)
            if boxes_conf_n > 0:
                boxes_xyxy = result.boxes.xyxy.tolist()
                boxes_cls = result.boxes.cls.tolist()

                for i, box_xyxy in enumerate(boxes_xyxy):
                    class_label_index = int(boxes_cls[i])  # 获取类别索引
                    modelname = name.get(class_label_index, "unknown")  # 获取物体名称
                    labellist.append(modelname)
                    xyxy = box_xyxy
                    left, top, right, bottom = map(int, xyxy)  # 获取边界框坐标
                    cv2.rectangle(
                        frame, (left, top), (right, bottom), (0, 255, 0), 1
                    )  # 绘制边界框
                    cv2.putText(
                        frame,
                        modelname,
                        (left, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        1,
                    )  # 标注物体名称

    # 添加FPS显示
    tend = time.time()
    fps = 1 / (tend - tstart)
    fps_text = "%.2f fps" % fps
    cv2.putText(
        frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1
    )

    # 将检测结果添加到返回列表
    ret.append(labellist)  # 添加检测到的物体名称列表
    ret.append(round(eye, 3))  # 添加眼睛纵横比，保留三位小数
    ret.append(round(mouth, 3))  # 添加嘴巴纵横比，保留三位小数

    # 返回处理后的结果和帧
    return ret, frame
