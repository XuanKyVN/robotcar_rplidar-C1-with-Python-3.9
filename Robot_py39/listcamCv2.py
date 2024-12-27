
#from cv2_enumerate_cameras import enumerate_cameras

#for camera_info in enumerate_cameras():
#    print(f'{camera_info.index}: {camera_info.name}')
'''
    1400: Integrated Webcam
1401: CyberLink YouCam 10
1402: HD Pro Webcam C920
700: Integrated Webcam
701: CyberLink YouCam 10
702: HD Pro Webcam C920
    '''

import cv2
from cv2_enumerate_cameras import enumerate_cameras

for camera_info in enumerate_cameras(cv2.CAP_MSMF):
    print(f'{camera_info.index}: {camera_info.name}')



