from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtGui import QPainter, QColor, QPolygon, QPaintEvent
from PySide6.QtCore import QPoint, QTimer

import math
import sys
import time


class VSyncUpdateWidget(QOpenGLWidget):

    @staticmethod
    def _create_channel_data(n_channels: int):
        data = []
        for i in range(n_channels):
            data.append(QPolygon([QPoint(x, 10 + 10 * i + math.sin(x / 10) * 10) for x in range(2000)]))
        return data

    def __init__(self, n_channels: int, parent: QWidget = None):
        super().__init__(parent)
        self.setAutoFillBackground(True)

        self._painter = QPainter()
        self._pen = QColor(255, 255, 255)

        self._last_paint_log_time = 0
        self._paint_calls = 0
        self._frames_per_second = 0

        self._data = VSyncUpdateWidget._create_channel_data(n_channels)

    def paintEvent(self, event: QPaintEvent) -> None:
        self.windowHandle().requestUpdate()  # request update on next VSync
        QTimer.singleShot(0, self._cycle_data)  # cycle data after paint done

        now = time.monotonic_ns()
        self._paint_calls += 1

        if self._last_paint_log_time == 0:
            self._last_paint_log_time = now
        else:
            secs_since_last_log = (now - self._last_paint_log_time) / 1e9
            if secs_since_last_log >= 1:
                self._frames_per_second = self._paint_calls / secs_since_last_log
                self._last_paint_log_time = now
                self._paint_calls = 0
                QTimer.singleShot(0, self._log_frames_per_seconds)

        self._painter.begin(self)
        self._painter.setPen(self._pen)

        for d in self._data:
            self._painter.drawPolyline(d)
        self._painter.end()

    def _cycle_data(self):
        for d in self._data:
            d.translate(-1, 0)
            f = d.front()
            d.removeFirst()
            f.setX(d.length())
            d.append(f)

    def _log_frames_per_seconds(self):
        print(f"Frames per second: {self._frames_per_second}")


my_application = QApplication(sys.argv)

my_widget = VSyncUpdateWidget(n_channels=12)
my_widget.show()

my_application.exec()
