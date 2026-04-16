import sys
import pyads
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QLineEdit, QGroupBox
)


class RobotUI(QWidget):
    def __init__(self):
        super().__init__()

        self.plc = None
        self.is_connected = False

        self.AMS_NET_ID = "169.254.189.18.1.1"
        self.AMS_PORT = 851

        self.init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_from_plc)
        self.timer.start(500)

    def init_ui(self):
        self.setWindowTitle("6-Axis Robot UI")
        self.resize(700, 600)

        # ── PLC Connection ──────────────────────────────────────────
        self.group_plc = QGroupBox("PLC Connection", self)
        self.group_plc.setGeometry(30, 30, 250, 110)

        self.btn_connect = QPushButton("Connect", self.group_plc)
        self.btn_connect.setGeometry(20, 30, 80, 30)
        self.btn_connect.clicked.connect(self.connect_plc)

        self.btn_disconnect = QPushButton("Disconnect", self.group_plc)
        self.btn_disconnect.setGeometry(110, 30, 90, 30)
        self.btn_disconnect.clicked.connect(self.disconnect_plc)

        self.btn_power_on = QPushButton("Power On", self.group_plc)
        self.btn_power_on.setGeometry(20, 70, 80, 30)
        self.btn_power_on.clicked.connect(self.power_on)

        self.label_plc_status = QLabel("Status: Disconnected", self.group_plc)
        self.label_plc_status.setGeometry(110, 75, 130, 25)

        self.label_ads = QLabel("ADS: Disconnected", self.group_plc)
        self.label_ads.setGeometry(20, 105, 200, 25)
        self.label_ads.setStyleSheet("color: gray")

        # ── Joint Control ───────────────────────────────────────────
        self.group_joint = QGroupBox("Joint Control", self)
        self.group_joint.setGeometry(30, 170, 360, 390)

        joints = [
            ("J1", 30),
            ("J2", 65),
            ("J3", 100),
            ("J4", 135),
            ("J5", 170),
            ("J6", 205),
        ]
        self.act_labels = {}
        self.inputs = {}
        self.move_btns = {}

        for name, y in joints:
            lbl = QLabel(f"{name} Act:", self.group_joint)
            lbl.setGeometry(20, y, 60, 25)

            act = QLabel("0.000", self.group_joint)
            act.setGeometry(80, y, 70, 25)
            self.act_labels[name] = act

            inp = QLineEdit(self.group_joint)
            inp.setGeometry(160, y, 70, 25)
            inp.setText("0")
            self.inputs[name] = inp

            btn = QPushButton("Move", self.group_joint)
            btn.setGeometry(250, y, 60, 25)
            self.move_btns[name] = btn

        # 綁定 Move 按鈕
        for i, name in enumerate(["J1", "J2", "J3", "J4", "J5", "J6"], start=1):
            idx = i
            self.move_btns[name].clicked.connect(
                lambda checked, n=name, x=idx: self.move_joint(n, x)
            )

        self.btn_move_all = QPushButton("Move All", self.group_joint)
        self.btn_move_all.setGeometry(110, 245, 100, 30)
        self.btn_move_all.clicked.connect(self.move_all_joints)

        # ── System Status ───────────────────────────────────────────
        self.group_status = QGroupBox("System Status", self)
        self.group_status.setGeometry(380, 30, 260, 140)

        self.label_status = QLabel("System: Ready", self.group_status)
        self.label_status.setGeometry(20, 30, 220, 25)

        self.label_mode = QLabel("Mode: Idle", self.group_status)
        self.label_mode.setGeometry(20, 65, 220, 25)

        self.label_error = QLabel("Error: None", self.group_status)
        self.label_error.setGeometry(20, 100, 220, 25)

    # ── PLC 連線 ────────────────────────────────────────────────────
    def connect_plc(self):
        try:
            self.plc = pyads.Connection(self.AMS_NET_ID, self.AMS_PORT)
            self.plc.open()
            self.is_connected = True

            self.label_plc_status.setText("Status: Connected")
            self.label_status.setText("System: PLC Connected")
            self.label_mode.setText("Mode: Standby")
            self.label_error.setText("Error: None")
            self.label_ads.setText("ADS: Connected")
            self.label_ads.setStyleSheet("color: green")

        except Exception as e:
            self.is_connected = False
            self.label_plc_status.setText("Status: Failed")
            self.label_error.setText(f"Error: {str(e)}")
            self.label_ads.setText("ADS: Error")
            self.label_ads.setStyleSheet("color: red")

    def disconnect_plc(self):
        try:
            if self.plc is not None:
                self.plc.close()
        except Exception:
            pass

        self.plc = None
        self.is_connected = False
        self.label_plc_status.setText("Status: Disconnected")
        self.label_status.setText("System: PLC Disconnected")
        self.label_mode.setText("Mode: Idle")
        self.label_ads.setText("ADS: Disconnected")
        self.label_ads.setStyleSheet("color: gray")

    # ── Power On ────────────────────────────────────────────────────
    def power_on(self):
        if not self.is_connected:
            self.label_error.setText("Error: PLC not connected")
            return
        try:
            self.plc.write_by_name("MAIN.HMI_PowerOn", True, pyads.PLCTYPE_BOOL)
            self.label_status.setText("System: Power On Command Sent")
            self.label_mode.setText("Mode: Power On")
            self.label_error.setText("Error: None")
        except Exception as e:
            self.label_error.setText(f"Error: {str(e)}")

    # ── 定時讀取實際位置 ────────────────────────────────────────────
    def update_from_plc(self):
        if not self.is_connected or self.plc is None:
            return
        try:
            for i, name in enumerate(["J1", "J2", "J3", "J4", "J5", "J6"], start=1):
                val = self.plc.read_by_name(f"MAIN.HMI_ActPos{i}", pyads.PLCTYPE_LREAL)
                self.act_labels[name].setText(f"{val:.3f}")

            self.label_status.setText("System: ADS Running")
            self.label_error.setText("Error: None")
        except Exception as e:
            self.label_status.setText("System: ADS Lost")
            self.label_error.setText(f"Error: {repr(e)}")

    # ── 單軸移動 ────────────────────────────────────────────────────
    def move_joint(self, name: str, idx: int):
        if not self.is_connected:
            self.label_error.setText("Error: PLC not connected")
            return
        try:
            target = float(self.inputs[name].text())
            self.plc.write_by_name(f"MAIN.HMI_MoveCmd{idx}", False, pyads.PLCTYPE_BOOL)
            self.plc.write_by_name(f"MAIN.HMI_TargetPos{idx}", target, pyads.PLCTYPE_LREAL)
            self.plc.write_by_name(f"MAIN.HMI_MoveCmd{idx}", True, pyads.PLCTYPE_BOOL)
            self.label_status.setText(f"System: {name} Move to {target}")
            self.label_error.setText("Error: None")
        except Exception as e:
            self.label_error.setText(f"Error: {repr(e)}")

    # ── 全軸移動 ────────────────────────────────────────────────────
    def move_all_joints(self):
        for i, name in enumerate(["J1", "J2", "J3", "J4", "J5", "J6"], start=1):
            self.move_joint(name, i)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotUI()
    window.show()
    sys.exit(app.exec())