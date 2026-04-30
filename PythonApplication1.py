# -*- coding: utf-8 -*-
import sys
import pyads
import threading
from PySide6.QtCore import QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QLineEdit, QGroupBox
)
from kinematics import get_end_effector_pose, inverse_kinematics_multistart


JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]


class IKSignals(QObject):
    finished = Signal(object, bool, object)


class RobotUI(QWidget):
    def __init__(self):
        super().__init__()

        self.plc           = None
        self.is_connected  = False
        self.ik_running    = False
        self._last_q       = None  # cache for FK update
        self.ik_signals    = IKSignals()
        self.ik_signals.finished.connect(self.on_ik_done)

        self.AMS_NET_ID = "169.254.189.18.1.1"
        self.AMS_PORT   = 851

        self.init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_from_plc)
        # timer started only after Connect

    def init_ui(self):
        self.setWindowTitle("6-Axis Robot UI")
        self.resize(780, 760)

        # ── PLC Connection ──────────────────────────────────────────
        self.group_plc = QGroupBox("PLC Connection", self)
        self.group_plc.setGeometry(30, 30, 250, 150)

        self.btn_connect = QPushButton("Connect", self.group_plc)
        self.btn_connect.setGeometry(20, 30, 80, 30)
        self.btn_connect.clicked.connect(self.connect_plc)

        self.btn_disconnect = QPushButton("Disconnect", self.group_plc)
        self.btn_disconnect.setGeometry(110, 30, 90, 30)
        self.btn_disconnect.clicked.connect(self.disconnect_plc)

        self.btn_power_on = QPushButton("Power On", self.group_plc)
        self.btn_power_on.setGeometry(20, 70, 80, 30)
        self.btn_power_on.clicked.connect(self.power_on)

        self.btn_reset = QPushButton("Reset", self.group_plc)
        self.btn_reset.setGeometry(110, 70, 90, 30)
        self.btn_reset.clicked.connect(self.reset_error)

        self.label_plc_status = QLabel("Status: Disconnected", self.group_plc)
        self.label_plc_status.setGeometry(20, 110, 220, 25)

        self.label_ads = QLabel("ADS: Disconnected", self.group_plc)
        self.label_ads.setGeometry(20, 130, 200, 18)
        self.label_ads.setStyleSheet("color: gray; font-size: 10pt;")

        # ── Joint Control ───────────────────────────────────────────
        self.group_joint = QGroupBox("Joint Control", self)
        self.group_joint.setGeometry(30, 200, 360, 280)

        self.act_labels = {}
        self.inputs     = {}
        self.move_btns  = {}

        for i, name in enumerate(JOINT_NAMES):
            y = 30 + i * 35
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

        for i, name in enumerate(JOINT_NAMES, start=1):
            self.move_btns[name].clicked.connect(
                lambda checked, n=name, x=i: self.move_joint(n, x)
            )

        self.btn_move_all = QPushButton("Move All", self.group_joint)
        self.btn_move_all.setGeometry(110, 245, 100, 30)
        self.btn_move_all.clicked.connect(self.move_all_joints)

        # ── System Status ───────────────────────────────────────────
        self.group_status = QGroupBox("System Status", self)
        self.group_status.setGeometry(420, 30, 320, 150)

        self.label_status = QLabel("System: Ready", self.group_status)
        self.label_status.setGeometry(20, 30, 280, 25)

        self.label_mode = QLabel("Mode: Idle", self.group_status)
        self.label_mode.setGeometry(20, 60, 280, 25)

        self.label_error = QLabel("Error: None", self.group_status)
        self.label_error.setGeometry(20, 90, 280, 50)
        self.label_error.setWordWrap(True)

        # ── FK ──────────────────────────────────────────────────────
        self.group_fk = QGroupBox(
            "Forward Kinematics (FK)  current joint -> end-effector pose", self)
        self.group_fk.setGeometry(30, 500, 720, 110)

        fk_axes = [("X",20),("Y",150),("Z",280),("Rz",410),("Ry",530),("Rx",650)]
        self.fk_labels = {}
        for axis, x in fk_axes:
            QLabel(f"{axis}:", self.group_fk).setGeometry(x, 30, 30, 25)
            val = QLabel("---", self.group_fk)
            val.setGeometry(x, 55, 110, 25)
            val.setStyleSheet("font-weight: bold; color: #0055aa;")
            self.fk_labels[axis] = val

        self.btn_fk = QPushButton("FK", self.group_fk)
        self.btn_fk.setGeometry(310, 80, 80, 22)
        self.btn_fk.clicked.connect(self.calc_fk)

        # ── IK ──────────────────────────────────────────────────────
        self.group_ik = QGroupBox(
            "Inverse Kinematics (IK)  target pose -> move arm", self)
        self.group_ik.setGeometry(30, 625, 720, 110)

        ik_axes = [("X",20),("Y",150),("Z",280),("Rz",410),("Ry",530),("Rx",650)]
        self.ik_inputs = {}
        for axis, x in ik_axes:
            QLabel(f"{axis}:", self.group_ik).setGeometry(x, 25, 30, 25)
            inp = QLineEdit(self.group_ik)
            inp.setGeometry(x, 50, 110, 25)
            inp.setText("0")
            self.ik_inputs[axis] = inp

        self.btn_ik = QPushButton("Calculate & Move", self.group_ik)
        self.btn_ik.setGeometry(290, 78, 130, 26)
        self.btn_ik.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold;"
        )
        self.btn_ik.clicked.connect(self.calc_ik_and_move)

    # ── PLC ─────────────────────────────────────────────────────────
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
            self.label_ads.setStyleSheet("color: green; font-size: 10pt;")
            self.timer.start(500)
        except Exception as e:
            self.is_connected = False
            self.label_plc_status.setText("Status: Failed")
            self.label_error.setText(f"Error: {str(e)}")
            self.label_ads.setText("ADS: Error")
            self.label_ads.setStyleSheet("color: red; font-size: 10pt;")

    def disconnect_plc(self):
        self.timer.stop()
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
        self.label_ads.setStyleSheet("color: gray; font-size: 10pt;")

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

    def reset_error(self):
        if not self.is_connected:
            self.label_error.setText("Error: PLC not connected")
            return
        try:
            # rising edge
            self.plc.write_by_name("MAIN.HMI_ResetCmd", False, pyads.PLCTYPE_BOOL)
            self.plc.write_by_name("MAIN.HMI_ResetCmd", True,  pyads.PLCTYPE_BOOL)
            self.label_status.setText("System: Reset Sent")
            self.label_error.setText("Error: None")
        except Exception as e:
            self.label_error.setText(f"Reset Error: {repr(e)}")

    # ── PLC polling ────────────────────────────────────────────────
    def update_from_plc(self):
        if not self.is_connected or self.plc is None:
            return
        try:
            for i, name in enumerate(JOINT_NAMES, start=1):
                val = self.plc.read_by_name(f"MAIN.HMI_ActPos{i}", pyads.PLCTYPE_LREAL)
                self.act_labels[name].setText(f"{val:.3f}")
            self.label_status.setText("System: ADS Running")
            self.label_error.setText("Error: None")

            # Update FK only when joint angles changed
            q_now = tuple(round(float(self.act_labels[n].text()), 3)
                          for n in JOINT_NAMES)
            if q_now != self._last_q:
                self._last_q = q_now
                self.calc_fk()
        except Exception as e:
            self.label_status.setText("System: ADS Lost")
            self.label_error.setText(f"Error: {repr(e)}")

    # ── Move ───────────────────────────────────────────────────────
    def move_joint(self, name, idx):
        if not self.is_connected:
            self.label_error.setText("Error: PLC not connected")
            return
        try:
            target = float(self.inputs[name].text())
            self.plc.write_by_name(f"MAIN.HMI_MoveCmd{idx}",   False, pyads.PLCTYPE_BOOL)
            self.plc.write_by_name(f"MAIN.HMI_TargetPos{idx}", target, pyads.PLCTYPE_LREAL)
            self.plc.write_by_name(f"MAIN.HMI_MoveCmd{idx}",   True,  pyads.PLCTYPE_BOOL)
            self.label_status.setText(f"System: {name} Move to {target}")
            self.label_error.setText("Error: None")
        except Exception as e:
            self.label_error.setText(f"Move Error: {repr(e)}")

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
            self.fk_labels["Rz"].setText(f"{euler[0]:.2f} deg")
            self.fk_labels["Ry"].setText(f"{euler[1]:.2f} deg")
            self.fk_labels["Rx"].setText(f"{euler[2]:.2f} deg")
        except Exception as e:
            self.label_error.setText(f"FK Error: {repr(e)}")

    # ── IK (background thread, hard-locked) ────────────────────────
    def calc_ik_and_move(self):
        if self.ik_running:
            return

        try:
            target_pos   = [float(self.ik_inputs[a].text()) for a in ["X","Y","Z"]]
            target_euler = [float(self.ik_inputs[a].text()) for a in ["Rz","Ry","Rx"]]
            q_init       = [float(self.act_labels[n].text()) for n in JOINT_NAMES]
        except Exception as e:
            self.label_error.setText(f"Input error: {repr(e)}")
            return

        self.ik_running = True
        self.btn_ik.setEnabled(False)
        self.btn_ik.setText("IK calculating...")
        self.btn_ik.setStyleSheet(
            "background-color: #888888; color: white; font-weight: bold;"
        )
        self.label_status.setText("System: IK calculating, please wait...")

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
        self.btn_ik.setText("Calculate & Move")
        self.btn_ik.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold;"
        )

        if success and q_result is not None:
            for i, name in enumerate(JOINT_NAMES):
                self.inputs[name].setText(f"{q_result[i]:.4f}")
            self.move_all_joints()
            self.label_status.setText(
                f"IK OK  pos_err={error[0]:.3f}mm  rot_err={error[1]:.3f}deg"
            )
            self.label_error.setText("Error: None")
        else:
            self.label_status.setText("System: IK Failed")
            self.label_error.setText(
                f"IK Failed  pos_err={error[0]:.2f}mm  rot_err={error[1]:.2f}deg"
            )

    # ── Cleanup on close ───────────────────────────────────────────
    def closeEvent(self, event):
        self.timer.stop()
        if self.is_connected:
            self.disconnect_plc()
        event.accept()


if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = RobotUI()
    window.show()
    sys.exit(app.exec())