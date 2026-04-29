# -*- coding: utf-8 -*-
import sys
import pyads
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QLineEdit, QGroupBox
)
from kinematics import get_end_effector_pose, inverse_kinematics_multistart


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
        self.resize(780, 720)

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
        self.group_joint.setGeometry(30, 170, 360, 280)

        joints = [("J1",30),("J2",65),("J3",100),("J4",135),("J5",170),("J6",205)]
        self.act_labels = {}
        self.inputs = {}
        self.move_btns = {}

        for name, y in joints:
            QLabel(f"{name} Act:", self.group_joint).setGeometry(20, y, 60, 25)

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

        for i, name in enumerate(["J1","J2","J3","J4","J5","J6"], start=1):
            self.move_btns[name].clicked.connect(
                lambda checked, n=name, x=i: self.move_joint(n, x)
            )

        self.btn_move_all = QPushButton("Move All", self.group_joint)
        self.btn_move_all.setGeometry(110, 245, 100, 30)
        self.btn_move_all.clicked.connect(self.move_all_joints)

        # ── System Status ───────────────────────────────────────────
        self.group_status = QGroupBox("System Status", self)
        self.group_status.setGeometry(420, 30, 320, 110)

        self.label_status = QLabel("System: Ready", self.group_status)
        self.label_status.setGeometry(20, 25, 280, 25)

        self.label_mode = QLabel("Mode: Idle", self.group_status)
        self.label_mode.setGeometry(20, 55, 280, 25)

        self.label_error = QLabel("Error: None", self.group_status)
        self.label_error.setGeometry(20, 85, 280, 25)

        # ── FK 正運動學顯示 ─────────────────────────────────────────
        self.group_fk = QGroupBox("Forward Kinematics (FK)  目前關節角 → 末端座標", self)
        self.group_fk.setGeometry(30, 470, 720, 110)

        fk_labels = [("X", 20), ("Y", 150), ("Z", 280), ("Rz", 410), ("Ry", 530), ("Rx", 650)]
        self.fk_labels = {}
        for axis, x in fk_labels:
            lbl = QLabel(f"{axis}:", self.group_fk)
            lbl.setGeometry(x, 30, 30, 25)
            val = QLabel("---", self.group_fk)
            val.setGeometry(x, 55, 110, 25)
            val.setStyleSheet("font-weight: bold; color: #0055aa;")
            self.fk_labels[axis] = val

        self.btn_fk = QPushButton("計算 FK", self.group_fk)
        self.btn_fk.setGeometry(300, 75, 100, 25)
        self.btn_fk.clicked.connect(self.calc_fk)

        # ── IK 逆運動學輸入 ─────────────────────────────────────────
        self.group_ik = QGroupBox("Inverse Kinematics (IK)  輸入目標座標 → 移動手臂", self)
        self.group_ik.setGeometry(30, 600, 720, 100)

        ik_axes = [("X", 20), ("Y", 150), ("Z", 280), ("Rz", 410), ("Ry", 530), ("Rx", 650)]
        self.ik_inputs = {}
        for axis, x in ik_axes:
            lbl = QLabel(f"{axis}:", self.group_ik)
            lbl.setGeometry(x, 25, 30, 25)
            inp = QLineEdit(self.group_ik)
            inp.setGeometry(x, 50, 110, 25)
            inp.setText("0")
            self.ik_inputs[axis] = inp

        self.btn_ik = QPushButton("計算並移動", self.group_ik)
        self.btn_ik.setGeometry(300, 65, 110, 28)
        self.btn_ik.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.btn_ik.clicked.connect(self.calc_ik_and_move)

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
            for i, name in enumerate(["J1","J2","J3","J4","J5","J6"], start=1):
                val = self.plc.read_by_name(f"MAIN.HMI_ActPos{i}", pyads.PLCTYPE_LREAL)
                self.act_labels[name].setText(f"{val:.3f}")
            self.label_status.setText("System: ADS Running")
            self.label_error.setText("Error: None")
            # 自動更新 FK 顯示
            self.calc_fk()
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
        for i, name in enumerate(["J1","J2","J3","J4","J5","J6"], start=1):
            self.move_joint(name, i)

    # ── FK 正運動學計算 ─────────────────────────────────────────────
    def calc_fk(self):
        try:
            q = [float(self.act_labels[n].text()) for n in ["J1","J2","J3","J4","J5","J6"]]
            pos, euler = get_end_effector_pose(q)
            self.fk_labels["X"].setText(f"{pos[0]:.2f} mm")
            self.fk_labels["Y"].setText(f"{pos[1]:.2f} mm")
            self.fk_labels["Z"].setText(f"{pos[2]:.2f} mm")
            self.fk_labels["Rz"].setText(f"{euler[0]:.2f} deg")
            self.fk_labels["Ry"].setText(f"{euler[1]:.2f} deg")
            self.fk_labels["Rx"].setText(f"{euler[2]:.2f} deg")
        except Exception as e:
            self.label_error.setText(f"FK Error: {repr(e)}")

    # ── IK 逆運動學計算並移動 ────────────────────────────────────────
    def calc_ik_and_move(self):
        try:
            target_pos   = [float(self.ik_inputs[a].text()) for a in ["X","Y","Z"]]
            target_euler = [float(self.ik_inputs[a].text()) for a in ["Rz","Ry","Rx"]]

            # 用目前關節角當初始猜測，加快收斂
            q_init = [float(self.act_labels[n].text()) for n in ["J1","J2","J3","J4","J5","J6"]]

            q_result, success, error = inverse_kinematics_multistart(
                target_pos=target_pos,
                target_euler_zyx_deg=target_euler,
                q_init_deg=q_init
            )

            if success:
                # 把 IK 結果填入 Joint Control 輸入欄
                for i, name in enumerate(["J1","J2","J3","J4","J5","J6"]):
                    self.inputs[name].setText(f"{q_result[i]:.4f}")

                # 移動全軸
                self.move_all_joints()
                self.label_status.setText(
                    f"IK OK  pos_err={error[0]:.3f}mm  rot_err={error[1]:.3f}deg"
                )
                self.label_error.setText("Error: None")
            else:
                self.label_error.setText(
                    f"IK Failed  pos_err={error[0]:.2f}mm  rot_err={error[1]:.2f}deg"
                )

        except Exception as e:
            self.label_error.setText(f"IK Error: {repr(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotUI()
    window.show()
    sys.exit(app.exec())