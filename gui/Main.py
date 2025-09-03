import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget
)

# 匯入 ML 與 RL 的主視窗類別
from MLP_GUI_C import MainWindowMLPC as MLPC
from DQN_GUI_C import MainWindowDQNC as DQNC
from MLP_GUI_S import MainWindowMLPS as MLPCS
from DQN_GUI_S import MainWindowDQNS as DQNS

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BGA-AI")
        self.setMinimumSize(400, 380)

        tab_widget = QTabWidget()
        tab_widget.addTab(MLPC(), "翹曲預測(凸)")
        tab_widget.addTab(MLPCS(), "翹曲預測(凹)")
        tab_widget.addTab(DQNC(), "AI設計(凸)")
        tab_widget.addTab(DQNS(), "AI設計(凹)")

        self.setCentralWidget(tab_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
