import numpy as np
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from robotcontrol import Ui_MainWindow  # chính là file chuyển từ .ui
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QMainWindow,QFileDialog,QMessageBox
import os, sys

class rplidarc1_Thread(QThread):
    change_pixmap_signal_rplidar = pyqtSignal(np.ndarray) # communicate to HMI

    def __init__(self):
        super().__init__()
        self._run_flag = True


    def run(self):  # this is Main program for running MQTT
        pass


    @pyqtSlot(dict)                                 # Receive signal from HMI
    def rplidar_receivedata(self, data):
        pass


    def stop(self):
            """Sets run flag to False and waits for thread to finish"""
            self._run_flag = False
            self.terminate()
            self.quit()
            # self.wait()




#-----------MAIN PROGRAM------------------------------------------
#---------------------------------------------------------------------
class MainWindow(QMainWindow):
    signal_to_rplidar_thread= pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)

        self.config_rplidar_thread()








    def config_rplidar_thread(self):

        #----------Config Serial Lora Thread--------
        self.rplidar = rplidarc1_Thread()
        self.change_pixmap_signal_rplidar.connect(self.receive_image_from_rplidar) # rplidar to HMI
        self.signal_to_rplidar_thread.connect(self.rplidar.rplidar_receivedata)  # HMI to rpLidar

        #self.lorathread.start()

    @pyqtSlot(np.ndarray)
    def receive_image_from_rplidar(self, cv_img):
        pass


    def start_rplidar(self):
        self.rplidar.start()
        self.flag_rplidar = True
        self.uic.plaintex_event.appendPlainText(str(datetime.now()) + ": " + "Start rplidar")

    def stop_rplidar(self):
        self.rplidar.stop()
        self.flag_rplidar = False
        self.uic.plaintex_event.appendPlainText(str(datetime.now()) + ": " + "Stop rplidar")



#---------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())