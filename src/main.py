import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from utils import load_model, predict_image


model = load_model("../models/best_model.pt") 

class AnimalClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hayvan Tanıma Uygulaması")
        self.setGeometry(200, 100, 700, 600)
        self.setStyleSheet("background-color: #f4f4f4;")

        self.image_label = QLabel("Henüz bir görüntü yüklenmedi")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(350, 350)
        self.image_label.setStyleSheet("""
            border: 2px dashed #aaa;
            background-color: #fff;
            color: #999;
            font-size: 14px;
        """)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 16))
        self.result_label.setStyleSheet("color: #007acc; margin-top: 10px;")

        self.button = QPushButton("Görüntü Yükle")
        self.button.setFixedHeight(40)
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border-radius: 8px;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #005999;
            }
        """)
        self.button.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Bir Görüntü Seç", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)

            predicted_name, confidence = predict_image(model, file_path)
            self.result_label.setText(
                f"<b>Tahmin:</b> {predicted_name.capitalize()}<br><b>Güven Skoru:</b> %{confidence*100:.2f}"
            )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnimalClassifierApp()
    window.show()
    sys.exit(app.exec_())
