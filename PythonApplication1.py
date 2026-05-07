# -*- coding: utf-8 -*-
import sys
import pyads
import threading
import numpy as np
from PySide6.QtCore import QTimer, Signal, QObject, Qt
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QBrush
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QLineEdit, QGroupBox
)
from kinematics import (
    get_end_effector_pose,
    inverse_kinematics_multistart,
    forward_kinematics,
    DH_PARAMS,# ← used by visualizer for accurate poses
)


# ── Cyberpunk 2077 palette ──────────────────────────────────────────
COLOR_BG       = "#0a0e14"
COLOR_BG_ALT   = "#11161e"
COLOR_PANEL    = "#0d1117"
COLOR_GRID     = "#11202c"
COLOR_BORDER   = "#1a3a4a"
COLOR_YELLOW   = "#fcee0a"
COLOR_CYAN     = "#00d9ff"
COLOR_RED      = "#ff003c"
COLOR_GREEN    = "#00ff9f"
COLOR_TEXT     = "#c5d4dd"
COLOR_TEXT_DIM = "#5a6878"

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]


CYBERPUNK_QSS = f"""
QWidget {{
    background-color: transparent;
    color: {COLOR_TEXT};
    font-family: 'Consolas', 'Courier New', 'Monaco', monospace;
    font-size: 9pt;
}}

QGroupBox {{
    background-color: rgba(13, 17, 23, 235);
    border: 1px solid {COLOR_CYAN};
    border-top: 2px solid {COLOR_CYAN};
    margin-top: 16px;
    padding-top: 14px;
    font-weight: bold;
    color: {COLOR_YELLOW};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 3px 14px;
    background-color: {COLOR_YELLOW};
    color: {COLOR_BG};
    font-weight: bold;
    font-size: 9pt;
}}

QPushButton {{
    background-color: {COLOR_BG_ALT};
    color: {COLOR_CYAN};
    border: 1px solid {COLOR_CYAN};
    padding: 3px 6px;
    font-weight: bold;
    font-size: 9pt;
}}
QPushButton:hover {{
    background-color: {COLOR_CYAN};
    color: {COLOR_BG};
}}
QPushButton:pressed {{
    background-color: {COLOR_YELLOW};
    color: {COLOR_BG};
    border: 1px solid {COLOR_YELLOW};
}}
QPushButton:disabled {{
    color: {COLOR_TEXT_DIM};
    border: 1px solid {COLOR_TEXT_DIM};
    background-color: {COLOR_PANEL};
}}

QPushButton#dangerBtn {{ color: {COLOR_RED}; border: 1px solid {COLOR_RED}; }}
QPushButton#dangerBtn:hover {{ background-color: {COLOR_RED}; color: {COLOR_BG}; }}

QPushButton#warnBtn {{ color: {COLOR_YELLOW}; border: 1px solid {COLOR_YELLOW}; }}
QPushButton#warnBtn:hover {{ background-color: {COLOR_YELLOW}; color: {COLOR_BG}; }}

QPushButton#primaryBtn {{
    background-color: {COLOR_YELLOW};
    color: {COLOR_BG};
    border: 1px solid {COLOR_YELLOW};
    font-weight: bold;
}}
QPushButton#primaryBtn:hover {{
    background-color: #ffffff;
    color: {COLOR_BG};
    border: 1px solid {COLOR_YELLOW};
}}
QPushButton#primaryBtn:pressed {{
    background-color: {COLOR_CYAN};
    border: 1px solid {COLOR_CYAN};
}}

QPushButton#calcBtn {{
    background-color: {COLOR_BG_ALT};
    color: {COLOR_TEXT_DIM};
    border: 1px dashed {COLOR_CYAN};
}}

QLineEdit {{
    background-color: {COLOR_BG};
    color: {COLOR_YELLOW};
    border: 1px solid {COLOR_BORDER};
    border-bottom: 2px solid {COLOR_CYAN};
    padding: 2px 4px;
    font-weight: bold;
    selection-background-color: {COLOR_YELLOW};
    selection-color: {COLOR_BG};
}}
QLineEdit:focus {{
    border: 1px solid {COLOR_YELLOW};
    border-bottom: 2px solid {COLOR_YELLOW};
    background-color: {COLOR_BG_ALT};
}}

QLabel {{ background: transparent; color: {COLOR_TEXT}; }}

QLabel#headerTitle {{
    color: {COLOR_YELLOW};
    font-weight: bold;
    font-size: 12pt;
    letter-spacing: 2px;
}}
QLabel#headerSub {{
    color: {COLOR_CYAN};
    font-size: 8pt;
    letter-spacing: 1px;
}}

QLabel#fkValue   {{ color: {COLOR_CYAN};   font-weight: bold; font-size: 10pt; }}
QLabel#actValue  {{ color: {COLOR_YELLOW}; font-weight: bold; font-size: 10pt; }}

QLabel#statusOk    {{ color: {COLOR_GREEN};    font-weight: bold; }}
QLabel#statusErr   {{ color: {COLOR_RED};      font-weight: bold; }}
QLabel#statusInfo  {{ color: {COLOR_CYAN};     font-weight: bold; }}
QLabel#statusDim   {{ color: {COLOR_TEXT_DIM}; font-weight: bold; }}

QLabel#fieldLabel {{
    color: {COLOR_TEXT_DIM};
    font-size: 8pt;
    font-weight: bold;
    letter-spacing: 1px;
}}
QLabel#axisLabel {{
    color: {COLOR_YELLOW};
    font-weight: bold;
    font-size: 9pt;
}}
"""


class IKSignals(QObject):
    finished = Signal(object, bool, object)


# ════════════════════════════════════════════════════════════════════
# 3D Arm Visualizer — uses kinematics.forward_kinematics for accuracy
# ════════════════════════════════════════════════════════════════════
# 把原本 robot_ui.py 裡整個 ArmVisualizer class 換成下面這版

JOINT_LABELS = ['BASE', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'TCP']

class ArmVisualizer(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.q = [0.0] * 6
        self.azimuth     = 35.0
        self.elevation   = 22.0
        self.auto_rotate = True
        self._dragging   = False
        self._last_pos   = None

        self.setCursor(Qt.OpenHandCursor)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def set_joint_angles(self, q_deg):
        try:
            self.q = [float(x) for x in q_deg]
        except Exception:
            pass

    def _tick(self):
        if self.auto_rotate:
            self.azimuth = (self.azimuth + 0.4) % 360
        self.update()

    def _joint_pivots(self):
        """
        旋轉軸實際位置(不是 frame 原點)。
        在 Modified Craig DH 裡:
            T_list[i].origin = Tz(d_i) 套完之後的點
            joint i 旋轉軸通過 = T_list[i].origin − T_list[i].Z 軸 × d_i
        這樣 J1 才會在地板,而不是被 d_1 推到肩膀。

        回傳 8 個點: BASE, J1, J2, J3, J4, J5, J6, TCP
        """
        _, T_list = forward_kinematics(self.q)
        pts = [T_list[0][:3, 3].copy()]            # BASE
        for i in range(6):
            d = DH_PARAMS[i][2]
            z_axis = T_list[i + 1][:3, 2]
            pivot  = T_list[i + 1][:3, 3] - z_axis * d
            pts.append(pivot)                       # J1..J6
        pts.append(T_list[6][:3, 3].copy())        # TCP
        return np.array(pts)

    def _project(self, pts):
        az = np.deg2rad(self.azimuth)
        el = np.deg2rad(self.elevation)
        Rz = np.array([
            [ np.cos(az),  np.sin(az), 0],
            [-np.sin(az),  np.cos(az), 0],
            [ 0,           0,          1],
        ])
        Rx = np.array([
            [1,  0,           0          ],
            [0,  np.cos(el),  np.sin(el) ],
            [0, -np.sin(el),  np.cos(el) ],
        ])
        return pts @ Rz.T @ Rx.T

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COLOR_BG))

        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            w, h = self.width(), self.height()

            pts = self._joint_pivots()              # 8 個 pivot 點
            tcp = pts[-1]

            if not np.all(np.isfinite(pts)):
                painter.setPen(QColor(COLOR_RED))
                painter.drawText(10, 20, "Invalid joint positions")
                return

            max_reach = max(float(np.max(np.abs(pts))), 100.0)
            scale = min(w, h) * 0.32 / max_reach
            cx, cy = w / 2.0, h / 2.0 + h * 0.08

            def to_screen(cam):
                sx = cam[..., 0] * scale + cx
                sy = -cam[..., 2] * scale + cy
                return sx, sy

            painter.setBrush(Qt.NoBrush)

            # ── grid ──
            gridn, gs = 6, max_reach / 4.0
            painter.setPen(QPen(QColor(COLOR_GRID), 1))
            for i in range(-gridn, gridn + 1):
                for line in (
                    [[-gridn*gs, i*gs, 0],[gridn*gs, i*gs, 0]],
                    [[i*gs, -gridn*gs, 0],[i*gs,  gridn*gs, 0]],
                ):
                    cam = self._project(np.array(line))
                    sx, sy = to_screen(cam)
                    painter.drawLine(int(sx[0]), int(sy[0]),
                                     int(sx[1]), int(sy[1]))

            # 圈
            ring = np.array([[np.cos(a)*max_reach*0.9,
                              np.sin(a)*max_reach*0.9, 0]
                             for a in np.linspace(0, 2*np.pi, 64)])
            cam_r = self._project(ring); sx_r, sy_r = to_screen(cam_r)
            painter.setPen(QPen(QColor(COLOR_BORDER), 1))
            for i in range(len(sx_r) - 1):
                painter.drawLine(int(sx_r[i]), int(sy_r[i]),
                                 int(sx_r[i+1]), int(sy_r[i+1]))

            # ── 世界座標軸 ──
            L = max_reach * 0.4
            for axis_pts, color in (
                (np.array([[0,0,0],[L,0,0]]), QColor(COLOR_RED)),
                (np.array([[0,0,0],[0,L,0]]), QColor(COLOR_GREEN)),
                (np.array([[0,0,0],[0,0,L]]), QColor(COLOR_CYAN)),
            ):
                cam_a = self._project(axis_pts)
                sx, sy = to_screen(cam_a)
                painter.setPen(QPen(color, 2))
                painter.drawLine(int(sx[0]), int(sy[0]),
                                 int(sx[1]), int(sy[1]))

            # ── project arm pivots ──
            cam_pts = self._project(pts)
            sx_all, sy_all = to_screen(cam_pts)

            # 把重疊的點合併成一個帶複合 label 的 marker
            merged = []   # (x, y, label_str, idx_set)
            for i, (x, y) in enumerate(zip(sx_all, sy_all)):
                xi, yi = int(x), int(y)
                placed = False
                for j, (mx, my, mlbl, idxs) in enumerate(merged):
                    if abs(xi - mx) < 6 and abs(yi - my) < 6:
                        merged[j] = (mx, my, mlbl + '/' + JOINT_LABELS[i],
                                     idxs | {i})
                        placed = True
                        break
                if not placed:
                    merged.append((xi, yi, JOINT_LABELS[i], {i}))

            # ── base 圓盤 ──
            bw = max(max_reach * scale * 0.18, 14)
            painter.setPen(QPen(QColor(COLOR_TEXT_DIM), 1))
            painter.setBrush(QColor("#1a1f29"))
            painter.drawEllipse(int(sx_all[0] - bw),
                                int(sy_all[0] - bw*0.3),
                                int(bw*2), int(bw*0.6))
            painter.setBrush(Qt.NoBrush)

            # ── 連桿 ──
            # 用合併後的點順序連線, 跳過長度為 0 的
            for k in range(len(merged) - 1):
                x1, y1 = merged[k][0], merged[k][1]
                x2, y2 = merged[k+1][0], merged[k+1][1]
                if abs(x1-x2) < 2 and abs(y1-y2) < 2:
                    continue
                # 大臂/小臂 cyan, 末端工具 yellow
                idxs_after = merged[k+1][3]
                tool_link = (7 in idxs_after)   # 7 = TCP
                color = QColor(COLOR_YELLOW) if tool_link else QColor(COLOR_CYAN)

                painter.setPen(QPen(QColor(0, 0, 0, 220), 7))
                painter.drawLine(x1, y1, x2, y2)
                glow = QColor(color); glow.setAlpha(70)
                painter.setPen(QPen(glow, 6))
                painter.drawLine(x1, y1, x2, y2)
                painter.setPen(QPen(color, 2.5))
                painter.drawLine(x1, y1, x2, y2)

            # ── 關節 marker + 文字 ──
            f_lbl = QFont("Consolas", 7); f_lbl.setBold(True)
            painter.setFont(f_lbl)

            for x, y, lbl, idxs in merged:
                is_base = (0 in idxs) and len(idxs) <= 2  # BASE 或 BASE/J1
                is_tcp  = (7 in idxs)

                if is_base:
                    painter.setPen(QPen(QColor(COLOR_GREEN), 2))
                    painter.setBrush(QColor(COLOR_BG))
                    painter.drawEllipse(x-5, y-5, 10, 10)
                    painter.setBrush(QColor(COLOR_GREEN))
                    painter.drawEllipse(x-2, y-2, 4, 4)
                    painter.setPen(QColor(COLOR_GREEN))
                    painter.drawText(x + 8, y + 4, lbl)
                elif is_tcp:
                    painter.setPen(QPen(QColor(COLOR_RED), 2))
                    painter.setBrush(QColor(COLOR_BG))
                    painter.drawEllipse(x-7, y-7, 14, 14)
                    painter.setBrush(QColor(COLOR_RED))
                    painter.drawEllipse(x-3, y-3, 6, 6)
                    painter.setPen(QPen(QColor(COLOR_RED), 1))
                    painter.drawLine(x-13, y, x-9, y)
                    painter.drawLine(x+9, y,  x+13, y)
                    painter.drawLine(x, y-13, x, y-9)
                    painter.drawLine(x, y+9,  x, y+13)
                    painter.setPen(QColor(COLOR_RED))
                    painter.drawText(x + 12, y + 4, lbl)
                else:
                    painter.setPen(QPen(QColor(COLOR_YELLOW), 2))
                    painter.setBrush(QColor(COLOR_BG))
                    painter.drawEllipse(x-5, y-5, 10, 10)
                    painter.setBrush(QColor(COLOR_YELLOW))
                    painter.drawEllipse(x-2, y-2, 4, 4)
                    painter.setPen(QColor(COLOR_YELLOW))
                    painter.drawText(x + 8, y + 4, lbl)

            painter.setBrush(Qt.NoBrush)

            # ── HUD ──
            f_small = QFont("Consolas", 7); f_small.setBold(True)
            painter.setFont(f_small)
            painter.setPen(QColor(COLOR_TEXT_DIM))
            painter.drawText(8, 14, f"AZ {self.azimuth:5.1f}°  EL {self.elevation:4.1f}°")

            if self.auto_rotate:
                painter.setPen(QColor(COLOR_GREEN))
                painter.drawText(8, 26, "▌AUTO")
            else:
                painter.setPen(QColor(COLOR_YELLOW))
                painter.drawText(8, 26, "▌MANUAL")

            painter.setPen(QColor(COLOR_CYAN))
            painter.drawText(w - 78, 14, "▣ ARM_SIM")

            f_tcp = QFont("Consolas", 8); f_tcp.setBold(True)
            painter.setFont(f_tcp)
            painter.setPen(QColor(COLOR_YELLOW))
            painter.drawText(8, h - 8,
                             f"TCP  X:{tcp[0]:7.1f}  Y:{tcp[1]:7.1f}  Z:{tcp[2]:7.1f} mm")

            painter.setPen(QPen(QColor(COLOR_CYAN), 1))
            painter.drawRect(0, 0, w-1, h-1)

        except Exception as e:
            import traceback
            traceback.print_exc()
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QColor(COLOR_RED))
            painter.drawText(10, 30, f"Viz error: {str(e)[:60]}")

    # mouse handlers — 跟原本一樣
    def mousePressEvent(self, event):
        self._dragging = True
        self.auto_rotate = False
        self._last_pos = event.position()
        self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        self._dragging = False
        self.setCursor(Qt.OpenHandCursor)

    def mouseMoveEvent(self, event):
        if not self._dragging or self._last_pos is None: return
        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self.azimuth = (self.azimuth + dx * 0.5) % 360
        self.elevation = max(-89.0, min(89.0, self.elevation + dy * 0.5))
        self._last_pos = pos

    def mouseDoubleClickEvent(self, event):
        self.auto_rotate = not self.auto_rotate


# ════════════════════════════════════════════════════════════════════
# Main UI
# ════════════════════════════════════════════════════════════════════
class RobotUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("mainWindow")

        self.plc           = None
        self.is_connected  = False
        self.ik_running    = False
        self._last_q       = None
        self.ik_signals    = IKSignals()
        self.ik_signals.finished.connect(self.on_ik_done)

        self.AMS_NET_ID = "169.254.189.18.1.1"
        self.AMS_PORT   = 851

        self._pulse_on = True

        self.init_ui()
        self.setStyleSheet(CYBERPUNK_QSS)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_from_plc)

        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self._toggle_pulse)
        self.pulse_timer.start(700)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QColor(COLOR_BG))
        painter.setBrush(Qt.NoBrush)   # safety: don't leak brush

        pen = QPen(QColor(COLOR_GRID))
        pen.setWidth(1)
        painter.setPen(pen)
        grid = 24
        for x in range(0, self.width(), grid):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), grid):
            painter.drawLine(0, y, self.width(), y)

        painter.setPen(QPen(QColor(COLOR_YELLOW), 2))
        c, m = 20, 4
        w, h = self.width(), self.height()
        painter.drawLine(m, m, m + c, m); painter.drawLine(m, m, m, m + c)
        painter.drawLine(w - m, m, w - m - c, m); painter.drawLine(w - m, m, w - m, m + c)
        painter.drawLine(m, h - m, m + c, h - m); painter.drawLine(m, h - m, m, h - m - c)
        painter.drawLine(w - m, h - m, w - m - c, h - m); painter.drawLine(w - m, h - m, w - m, h - m - c)

        painter.setPen(QPen(QColor(COLOR_CYAN), 1))
        painter.drawLine(20, 64, w - 20, 64)
        painter.setPen(QPen(QColor(COLOR_YELLOW), 2))
        painter.drawLine(20, 66, 110, 66)

    def _toggle_pulse(self):
        self._pulse_on = not self._pulse_on
        if self.is_connected:
            color = COLOR_GREEN if self._pulse_on else "#004422"
        else:
            color = COLOR_RED if self._pulse_on else "#440011"
        self.pulse_dot.setStyleSheet(
            f"color: {color}; font-weight: bold; font-size: 18pt; background: transparent;"
        )

    def init_ui(self):
        self.setWindowTitle("NEURAL_LINK :: 6-AXIS ROBOTIC CONTROL")
        self.resize(800, 830)

        # ── Header ─────────────────────────────────────────────────
        self.header_title = QLabel("▣ NEURAL_LINK // 6-AXIS ROBOTIC CONTROL UNIT", self)
        self.header_title.setObjectName("headerTitle")
        self.header_title.setGeometry(20, 14, 720, 24)

        self.header_sub = QLabel(
            "◢ SYS.v2.077    ◢ ADS PROTOCOL    ◢ KINEMATICS_ENGINE: ONLINE", self)
        self.header_sub.setObjectName("headerSub")
        self.header_sub.setGeometry(20, 40, 720, 16)

        self.pulse_dot = QLabel("●", self)
        self.pulse_dot.setGeometry(760, 18, 24, 28)
        self.pulse_dot.setStyleSheet(
            f"color: {COLOR_RED}; font-weight: bold; font-size: 18pt; background: transparent;"
        )

        # ── PLC Connection ─────────────────────────────────────────
        self.group_plc = QGroupBox("[ CON.NET ] PLC CONNECTION", self)
        self.group_plc.setGeometry(20, 90, 380, 170)

        self.btn_connect = QPushButton("CONNECT", self.group_plc)
        self.btn_connect.setGeometry(15, 35, 170, 28)
        self.btn_connect.clicked.connect(self.connect_plc)

        self.btn_disconnect = QPushButton("DISCONNECT", self.group_plc)
        self.btn_disconnect.setGeometry(195, 35, 170, 28)
        self.btn_disconnect.clicked.connect(self.disconnect_plc)

        self.btn_power_on = QPushButton("⚡ PWR ON", self.group_plc)
        self.btn_power_on.setObjectName("warnBtn")
        self.btn_power_on.setGeometry(15, 72, 170, 28)
        self.btn_power_on.clicked.connect(self.power_on)

        self.btn_reset = QPushButton("⟲ RESET", self.group_plc)
        self.btn_reset.setObjectName("dangerBtn")
        self.btn_reset.setGeometry(195, 72, 170, 28)
        self.btn_reset.clicked.connect(self.reset_error)

        l1 = QLabel("◆ STATUS", self.group_plc); l1.setObjectName("fieldLabel")
        l1.setGeometry(15, 110, 80, 16)
        self.label_plc_status = QLabel("DISCONNECTED", self.group_plc)
        self.label_plc_status.setObjectName("statusDim")
        self.label_plc_status.setGeometry(95, 110, 270, 16)

        l2 = QLabel("◆ ADS", self.group_plc); l2.setObjectName("fieldLabel")
        l2.setGeometry(15, 132, 80, 16)
        self.label_ads = QLabel("OFFLINE", self.group_plc)
        self.label_ads.setObjectName("statusDim")
        self.label_ads.setGeometry(95, 132, 270, 16)

        # ── Joint Control ──────────────────────────────────────────
        self.group_joint = QGroupBox("[ JOINT.CTRL ] JOINT CONTROL", self)
        self.group_joint.setGeometry(20, 275, 380, 305)

        for tx, lbl in [(20,"AXIS"), (80,"CURRENT"), (170,"TARGET")]:
            hdr = QLabel(lbl, self.group_joint); hdr.setObjectName("fieldLabel")
            hdr.setGeometry(tx, 30, 70, 16)

        self.act_labels = {}
        self.inputs     = {}
        self.move_btns  = {}

        for i, name in enumerate(JOINT_NAMES):
            y = 52 + i * 36
            tag = QLabel(f"▸ {name}", self.group_joint)
            tag.setObjectName("axisLabel")
            tag.setGeometry(20, y, 55, 25)

            act = QLabel("0.000", self.group_joint)
            act.setObjectName("actValue")
            act.setGeometry(80, y, 75, 25)
            self.act_labels[name] = act

            inp = QLineEdit(self.group_joint)
            inp.setGeometry(165, y, 80, 25)
            inp.setText("0")
            self.inputs[name] = inp

            btn = QPushButton("▶ MOVE", self.group_joint)
            btn.setGeometry(255, y, 100, 25)
            self.move_btns[name] = btn

        for i, name in enumerate(JOINT_NAMES, start=1):
            self.move_btns[name].clicked.connect(
                lambda checked, n=name, x=i: self.move_joint(n, x)
            )

        self.btn_move_all = QPushButton("▶▶  MOVE ALL", self.group_joint)
        self.btn_move_all.setObjectName("primaryBtn")
        self.btn_move_all.setGeometry(115, 268, 150, 28)
        self.btn_move_all.clicked.connect(self.move_all_joints)

        # ── System Status ──────────────────────────────────────────
        self.group_status = QGroupBox("[ SYS.STATUS ] SYSTEM STATUS", self)
        self.group_status.setGeometry(420, 90, 360, 170)

        ls = QLabel("◆ SYSTEM", self.group_status); ls.setObjectName("fieldLabel")
        ls.setGeometry(15, 32, 80, 16)
        self.label_status = QLabel("READY", self.group_status)
        self.label_status.setObjectName("statusInfo")
        self.label_status.setGeometry(100, 32, 250, 16)

        lm = QLabel("◆ MODE", self.group_status); lm.setObjectName("fieldLabel")
        lm.setGeometry(15, 60, 80, 16)
        self.label_mode = QLabel("IDLE", self.group_status)
        self.label_mode.setObjectName("statusDim")
        self.label_mode.setGeometry(100, 60, 250, 16)

        le = QLabel("◆ ERROR", self.group_status); le.setObjectName("fieldLabel")
        le.setGeometry(15, 90, 80, 16)
        self.label_error = QLabel("NONE", self.group_status)
        self.label_error.setObjectName("statusOk")
        self.label_error.setGeometry(100, 90, 250, 60)
        self.label_error.setWordWrap(True)
        self.label_error.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # ── Arm Visualizer ─────────────────────────────────────────
        self.group_arm = QGroupBox(
            "[ ARM.SIM ] LIVE 3D VIEW  ::  drag = orbit  /  dbl-click = auto",
            self)
        self.group_arm.setGeometry(420, 275, 360, 305)

        self.arm_view = ArmVisualizer(self.group_arm)
        self.arm_view.setGeometry(8, 25, 344, 272)

        # ── FK ─────────────────────────────────────────────────────
        self.group_fk = QGroupBox(
            "[ FK.MODULE ] FORWARD KINEMATICS  ::  joint → end-effector pose", self)
        self.group_fk.setGeometry(20, 595, 760, 110)

        fk_axes = [("X",20),("Y",150),("Z",280),("Rz",410),("Ry",540),("Rx",670)]
        self.fk_labels = {}
        for axis, x in fk_axes:
            tag = QLabel(f"◢ {axis}", self.group_fk); tag.setObjectName("axisLabel")
            tag.setGeometry(x, 30, 50, 20)
            val = QLabel("---", self.group_fk); val.setObjectName("fkValue")
            val.setGeometry(x, 52, 120, 22)
            self.fk_labels[axis] = val

        self.btn_fk = QPushButton("▶ FK SOLVE", self.group_fk)
        self.btn_fk.setGeometry(330, 80, 110, 24)
        self.btn_fk.clicked.connect(self.calc_fk)

        # ── IK ─────────────────────────────────────────────────────
        self.group_ik = QGroupBox(
            "[ IK.MODULE ] INVERSE KINEMATICS  ::  target pose → move arm", self)
        self.group_ik.setGeometry(20, 720, 760, 100)

        ik_axes = [("X",20),("Y",150),("Z",280),("Rz",410),("Ry",540),("Rx",670)]
        self.ik_inputs = {}
        for axis, x in ik_axes:
            tag = QLabel(f"◢ {axis}", self.group_ik); tag.setObjectName("axisLabel")
            tag.setGeometry(x, 25, 50, 20)
            inp = QLineEdit(self.group_ik); inp.setGeometry(x, 47, 120, 24)
            inp.setText("0")
            self.ik_inputs[axis] = inp

        self.btn_ik = QPushButton("▶ EXEC // IK SOLVE", self.group_ik)
        self.btn_ik.setObjectName("primaryBtn")
        self.btn_ik.setGeometry(305, 75, 170, 22)
        self.btn_ik.clicked.connect(self.calc_ik_and_move)

    # ── helpers ────────────────────────────────────────────────────
    def _set_status(self, label, text, kind="info"):
        names = {"ok":"statusOk","err":"statusErr","info":"statusInfo","dim":"statusDim"}
        label.setObjectName(names.get(kind, "statusInfo"))
        label.setText(text)
        label.style().unpolish(label); label.style().polish(label)

    def _repolish(self, widget):
        widget.style().unpolish(widget); widget.style().polish(widget)

    # ── PLC ────────────────────────────────────────────────────────
    def connect_plc(self):
        try:
            self.plc = pyads.Connection(self.AMS_NET_ID, self.AMS_PORT)
            self.plc.open()
            self.is_connected = True
            self._set_status(self.label_plc_status, "● CONNECTED", "ok")
            self._set_status(self.label_status, "PLC LINK ESTABLISHED", "ok")
            self._set_status(self.label_mode, "STANDBY", "info")
            self._set_status(self.label_error, "NONE", "ok")
            self._set_status(self.label_ads, "● ONLINE", "ok")
            self.timer.start(500)
        except Exception as e:
            self.is_connected = False
            self._set_status(self.label_plc_status, "✕ FAILED", "err")
            self._set_status(self.label_error, f"{str(e)}", "err")
            self._set_status(self.label_ads, "✕ ERROR", "err")

    def disconnect_plc(self):
        self.timer.stop()
        try:
            if self.plc is not None:
                self.plc.close()
        except Exception:
            pass
        self.plc = None
        self.is_connected = False
        self._set_status(self.label_plc_status, "DISCONNECTED", "dim")
        self._set_status(self.label_status, "PLC OFFLINE", "dim")
        self._set_status(self.label_mode, "IDLE", "dim")
        self._set_status(self.label_ads, "OFFLINE", "dim")

    def power_on(self):
        if not self.is_connected:
            self._set_status(self.label_error, "PLC NOT CONNECTED", "err"); return
        try:
            self.plc.write_by_name("MAIN.HMI_PowerOn", True, pyads.PLCTYPE_BOOL)
            self._set_status(self.label_status, "POWER ON :: CMD SENT", "info")
            self._set_status(self.label_mode, "POWER ON", "ok")
            self._set_status(self.label_error, "NONE", "ok")
        except Exception as e:
            self._set_status(self.label_error, f"{str(e)}", "err")

    def reset_error(self):
        if not self.is_connected:
            self._set_status(self.label_error, "PLC NOT CONNECTED", "err"); return
        try:
            self.plc.write_by_name("MAIN.HMI_ResetCmd", False, pyads.PLCTYPE_BOOL)
            self.plc.write_by_name("MAIN.HMI_ResetCmd", True,  pyads.PLCTYPE_BOOL)
            self._set_status(self.label_status, "RESET :: CMD SENT", "info")
            self._set_status(self.label_error, "NONE", "ok")
        except Exception as e:
            self._set_status(self.label_error, f"RESET ERR :: {repr(e)}", "err")

    # ── PLC polling ────────────────────────────────────────────────
    def update_from_plc(self):
        if not self.is_connected or self.plc is None:
            return
        try:
            for i, name in enumerate(JOINT_NAMES, start=1):
                val = self.plc.read_by_name(f"MAIN.HMI_ActPos{i}", pyads.PLCTYPE_LREAL)
                self.act_labels[name].setText(f"{val:.3f}")
            self._set_status(self.label_status, "ADS :: STREAMING", "ok")
            self._set_status(self.label_error, "NONE", "ok")

            q = [float(self.act_labels[n].text()) for n in JOINT_NAMES]
            self.arm_view.set_joint_angles(q)            # ← live arm sync

            q_now = tuple(round(v, 3) for v in q)
            if q_now != self._last_q:
                self._last_q = q_now
                self.calc_fk()
        except Exception as e:
            self._set_status(self.label_status, "ADS :: SIGNAL LOST", "err")
            self._set_status(self.label_error, f"{repr(e)}", "err")

    # ── Move ───────────────────────────────────────────────────────
    def move_joint(self, name, idx):
        if not self.is_connected:
            self._set_status(self.label_error, "PLC NOT CONNECTED", "err"); return
        try:
            target = float(self.inputs[name].text())
            self.plc.write_by_name(f"MAIN.HMI_MoveCmd{idx}",   False, pyads.PLCTYPE_BOOL)
            self.plc.write_by_name(f"MAIN.HMI_TargetPos{idx}", target, pyads.PLCTYPE_LREAL)
            self.plc.write_by_name(f"MAIN.HMI_MoveCmd{idx}",   True,  pyads.PLCTYPE_BOOL)
            self._set_status(self.label_status, f"{name} ▶ {target}", "info")
            self._set_status(self.label_error, "NONE", "ok")
        except Exception as e:
            self._set_status(self.label_error, f"MOVE ERR :: {repr(e)}", "err")

    def move_all_joints(self):
        for i, name in enumerate(JOINT_NAMES, start=1):
            self.move_joint(name, i)

    # ── FK ─────────────────────────────────────────────────────────
    def calc_fk(self):
        try:
            q = [float(self.act_labels[n].text()) for n in JOINT_NAMES]
            pos, euler = get_end_effector_pose(q)
            self.fk_labels["X"].setText(f"{pos[0]:.2f} mm")
            self.fk_labels["Y"].setText(f"{pos[1]:.2f} mm")
            self.fk_labels["Z"].setText(f"{pos[2]:.2f} mm")
            self.fk_labels["Rz"].setText(f"{euler[0]:.2f}°")
            self.fk_labels["Ry"].setText(f"{euler[1]:.2f}°")
            self.fk_labels["Rx"].setText(f"{euler[2]:.2f}°")
        except Exception as e:
            self._set_status(self.label_error, f"FK ERR :: {repr(e)}", "err")

    # ── IK ─────────────────────────────────────────────────────────
    def calc_ik_and_move(self):
        if self.ik_running:
            return

        try:
            target_pos   = [float(self.ik_inputs[a].text()) for a in ["X","Y","Z"]]
            target_euler = [float(self.ik_inputs[a].text()) for a in ["Rz","Ry","Rx"]]
            q_init       = [float(self.act_labels[n].text()) for n in JOINT_NAMES]
        except Exception as e:
            self._set_status(self.label_error, f"INPUT ERR :: {repr(e)}", "err")
            return

        self.ik_running = True
        self.btn_ik.setEnabled(False)
        self.btn_ik.setText("◌ COMPUTING...")
        self.btn_ik.setObjectName("calcBtn")
        self._repolish(self.btn_ik)
        self._set_status(self.label_status, "IK :: COMPUTING [ HOLD ]", "info")

        def run_ik():
            try:
                q_result, success, error = inverse_kinematics_multistart(
                    target_pos=target_pos,
                    target_euler_zyx_deg=target_euler,
                    q_init_deg=q_init
                )
            except Exception:
                q_result, success, error = None, False, (999.0, 999.0)
            self.ik_signals.finished.emit(q_result, success, error)

        threading.Thread(target=run_ik, daemon=True).start()

    def on_ik_done(self, q_result, success, error):
        self.ik_running = False
        self.btn_ik.setEnabled(True)
        self.btn_ik.setText("▶ EXEC // IK SOLVE")
        self.btn_ik.setObjectName("primaryBtn")
        self._repolish(self.btn_ik)

        if success and q_result is not None:
            for i, name in enumerate(JOINT_NAMES):
                self.inputs[name].setText(f"{q_result[i]:.4f}")
            self.move_all_joints()
            self._set_status(self.label_status,
                f"IK ✓ OK   pos_err={error[0]:.3f}mm   rot_err={error[1]:.3f}°", "ok")
            self._set_status(self.label_error, "NONE", "ok")
        else:
            self._set_status(self.label_status, "IK ✕ FAILED", "err")
            self._set_status(self.label_error,
                f"IK FAILED  pos_err={error[0]:.2f}mm  rot_err={error[1]:.2f}°", "err")

    # ── Cleanup ────────────────────────────────────────────────────
    def closeEvent(self, event):
        self.timer.stop()
        self.pulse_timer.stop()
        if hasattr(self, 'arm_view'):
            self.arm_view.timer.stop()
        if self.is_connected:
            self.disconnect_plc()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = RobotUI()
    window.show()
    sys.exit(app.exec())