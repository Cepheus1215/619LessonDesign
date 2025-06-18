import sys
import os
import csv
import torch
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QTextEdit, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, exposure, transform
import cv2
import random
import warnings

# 忽略所有FutureWarning和DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# 定义交通标志识别CNN模型（与训练代码一致）
class TrafficSignNet(nn.Module):
    def __init__(self, width, height, depth, classes):
        super(TrafficSignNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=depth, out_channels=8,
                               kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16,
                               kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * (height // 8) * (width // 8), 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, classes)

        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        x = torch.flatten(x, 1)

        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class DetectionThread(QThread):
    """后台检测线程"""
    finished = pyqtSignal(np.ndarray, str)
    error = pyqtSignal(str)

    def __init__(self, image_path, model, class_names):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.class_names = class_names

    def run(self):
        try:
            # 加载并预处理图像
            image = io.imread(self.image_path)
            original_image = image.copy()
            image = transform.resize(image, (32, 32))
            image = exposure.equalize_adapthist(image, clip_limit=0.1)
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

            # 模型推理
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.exp(outputs)  # 转换为概率
                confidence, preds = torch.max(probabilities, 1)

            # 获取结果
            class_idx = preds.item()
            confidence_val = confidence.item()
            class_name = self.class_names[class_idx]

            # 在图像上绘制结果
            result_img = cv2.resize(original_image, (400, 400))

            # 计算文本位置
            text = f"{class_name}: {confidence_val:.2f}"
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # 绘制半透明背景
            overlay = result_img.copy()
            cv2.rectangle(overlay, (10, 10), (20 + text_width, 40 + text_height), (0, 0, 0), -1)
            alpha = 0.6
            result_img = cv2.addWeighted(overlay, alpha, result_img, 1 - alpha, 0)

            # 绘制文本
            cv2.putText(result_img, text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 255, 0), thickness, cv2.LINE_AA)

            # 准备结果文本
            result_text = f"检测结果: {class_name} (置信度: {confidence_val:.2f})"

            self.finished.emit(result_img, result_text)

        except Exception as e:
            error_msg = f"检测过程中发生错误:\n{str(e)}"
            self.error.emit(error_msg)


class TrafficSignApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交通标志识别系统")
        self.setGeometry(100, 100, 1000, 700)

        # 创建主部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 标题
        title_label = QLabel("交通标志识别系统")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 15px;")
        main_layout.addWidget(title_label)

        # 创建图像显示区域
        image_layout = QHBoxLayout()

        # 原始图像面板
        original_panel = QVBoxLayout()
        original_label = QLabel("原始图像")
        original_label.setFont(QFont("Arial", 12, QFont.Bold))
        original_label.setAlignment(Qt.AlignCenter)
        self.original_image = QLabel()
        self.original_image.setAlignment(Qt.AlignCenter)
        self.original_image.setMinimumSize(400, 400)
        self.original_image.setStyleSheet("border: 2px solid #3498db; background-color: #f8f9fa;")
        original_panel.addWidget(original_label)
        original_panel.addWidget(self.original_image)

        # 结果图像面板
        result_panel = QVBoxLayout()
        result_label = QLabel("检测结果")
        result_label.setFont(QFont("Arial", 12, QFont.Bold))
        result_label.setAlignment(Qt.AlignCenter)
        self.result_image = QLabel()
        self.result_image.setAlignment(Qt.AlignCenter)
        self.result_image.setMinimumSize(400, 400)
        self.result_image.setStyleSheet("border: 2px solid #2ecc71; background-color: #f8f9fa;")
        result_panel.addWidget(result_label)
        result_panel.addWidget(self.result_image)

        image_layout.addLayout(original_panel)
        image_layout.addLayout(result_panel)
        main_layout.addLayout(image_layout)

        # 信息显示区域
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QFont("Arial", 10))
        self.info_text.setStyleSheet("background-color: #f8f9fa; padding: 10px; border: 1px solid #ddd;")
        main_layout.addWidget(self.info_text)

        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        # 创建按钮
        self.load_model_btn = self.create_button("加载模型", "#3498db")
        self.load_class_btn = self.create_button("加载类别", "#2ecc71")
        self.load_image_btn = self.create_button("加载图像", "#e74c3c")
        self.detect_btn = self.create_button("开始识别", "#f39c12")
        self.clear_btn = self.create_button("清除结果", "#95a5a6")

        # 添加按钮到布局
        button_layout.addWidget(self.load_model_btn)
        button_layout.addWidget(self.load_class_btn)
        button_layout.addWidget(self.load_image_btn)
        button_layout.addWidget(self.detect_btn)
        button_layout.addWidget(self.clear_btn)

        main_layout.addLayout(button_layout)

        # 连接信号槽
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_class_btn.clicked.connect(self.load_class_names)
        self.load_image_btn.clicked.connect(self.load_image)
        self.detect_btn.clicked.connect(self.detect_signs)
        self.clear_btn.clicked.connect(self.clear_results)

        # 初始化变量
        self.model = None
        self.image_path = None
        self.class_names = {}
        self.result_img = None

        # 初始状态设置
        self.detect_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        # 添加欢迎信息
        self.info_text.append("=== 交通标志识别系统 ===")
        self.info_text.append("使用方法:")
        self.info_text.append("1. 点击'加载模型'选择训练好的.pth文件")
        self.info_text.append("2. 点击'加载类别'选择signnames.csv文件")
        self.info_text.append("3. 点击'加载图像'选择要识别的交通标志图片")
        self.info_text.append("4. 点击'开始识别'进行交通标志识别")
        self.info_text.append("")
        self.info_text.append("注意: 请先加载模型和类别文件，然后加载图像进行识别")

    def create_button(self, text, color):
        """创建样式化按钮"""
        button = QPushButton(text)
        button.setFont(QFont("Arial", 10, QFont.Bold))
        button.setFixedHeight(40)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: {self.darken_color(color)};
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
                color: #7f8c8d;
            }}
        """)
        button.setCursor(Qt.PointingHandCursor)
        return button

    def darken_color(self, hex_color, factor=0.8):
        """使颜色变暗"""
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = max(0, min(255, int(r * factor)))
        g = max(0, min(255, int(g * factor)))
        b = max(0, min(255, int(b * factor)))
        return f"#{r:02x}{g:02x}{b:02x}"

    def load_class_names(self):
        """加载交通标志类别名称"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择类别文件", "", "CSV文件 (*.csv)")

        if not file_path:
            return

        try:
            # 清空当前类别
            self.class_names = {}

            # 读取CSV文件
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    class_id = int(row['ClassId'])
                    sign_name = row['SignName']
                    self.class_names[class_id] = sign_name

            # 更新UI状态
            self.info_text.append(f"✅ 类别加载成功: {os.path.basename(file_path)}")
            self.info_text.append(f"已加载 {len(self.class_names)} 个交通标志类别")

            # 显示前5个类别作为示例
            self.info_text.append("示例类别:")
            for i in range(min(5, len(self.class_names))):
                self.info_text.append(f"  {i}: {self.class_names[i]}")

            # 如果模型已加载，启用检测按钮
            if self.model and self.image_path:
                self.detect_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "类别加载错误",
                                 f"无法加载类别文件:\n{str(e)}\n\n"
                                 "请确保文件格式正确且包含ClassId和SignName列")

    def load_model(self):
        """加载模型"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch模型 (*.pth)")

        if not model_path:
            return

        try:
            # 创建模型实例
            self.model = TrafficSignNet(width=32, height=32, depth=3, classes=43)

            # 加载状态字典
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()

            # 更新UI状态
            self.info_text.append(f"✅ 模型加载成功: {os.path.basename(model_path)}")
            self.info_text.append(f"模型结构: {type(self.model).__name__}")

            # 如果类别已加载且图像存在，启用检测按钮
            if self.class_names and self.image_path:
                self.detect_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "模型加载错误",
                                 f"无法加载模型:\n{str(e)}\n\n"
                                 "请确保模型文件与当前应用兼容")
            self.model = None
            self.detect_btn.setEnabled(False)

    def load_image(self):
        """加载待识别的图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            try:
                self.image_path = file_path

                # 显示原始图像
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    self.original_image.setPixmap(
                        pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

                    # 清除之前的检测结果
                    self.result_image.clear()
                    self.info_text.append(f"✅ 图像加载成功: {os.path.basename(file_path)}")

                    # 如果模型和类别已加载，启用检测按钮
                    if self.model and self.class_names:
                        self.detect_btn.setEnabled(True)
                        self.clear_btn.setEnabled(True)
                else:
                    QMessageBox.warning(self, "图像错误", "无法加载选定的图像文件")

            except Exception as e:
                QMessageBox.critical(self, "图像加载错误",
                                     f"无法加载图像:\n{str(e)}")

    def detect_signs(self):
        """启动检测线程"""
        if not self.image_path:
            QMessageBox.warning(self, "警告", "请先加载图像!")
            return

        if not self.model:
            QMessageBox.warning(self, "警告", "请先加载模型!")
            return

        if not self.class_names:
            QMessageBox.warning(self, "警告", "请先加载交通标志类别!")
            return

        # 禁用按钮防止重复点击
        self.detect_btn.setEnabled(False)
        self.info_text.append("开始识别交通标志...")

        # 创建并启动检测线程
        self.thread = DetectionThread(self.image_path, self.model, self.class_names)
        self.thread.finished.connect(self.handle_detection_result)
        self.thread.error.connect(self.handle_detection_error)
        self.thread.start()

    def handle_detection_result(self, result_img, result_text):
        """处理检测成功结果"""
        # 转换并显示结果图像
        height, width, channel = result_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(result_img.data, width, height,
                       bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.result_image.setPixmap(pixmap)

        # 保存结果图像
        self.result_img = result_img

        # 显示结果信息
        self.info_text.append(result_text)
        self.info_text.append("✅ 识别完成!")

        # 启用按钮
        self.detect_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

    def handle_detection_error(self, error_msg):
        """处理检测错误"""
        QMessageBox.critical(self, "识别错误", error_msg)
        self.info_text.append("❌ 识别失败!")
        self.detect_btn.setEnabled(True)

    def clear_results(self):
        """清除识别结果"""
        self.result_image.clear()
        self.info_text.clear()
        self.result_img = None
        self.info_text.append("结果已清除")
        self.info_text.append("可以加载新的图像进行识别")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")

    window = TrafficSignApp()
    window.show()
    sys.exit(app.exec_())