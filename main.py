import RPi.GPIO as GPIO
import time
import cv2
from level import measure_level
from flow import FlowSensor
from tank_detection_v2 import TankDetection
import pandas as pd

GPIO.setmode(GPIO.BCM)
REAL_TANK_HEIGHT_CM = 28.5	# tank height in cm
detect = TankDetection(REAL_TANK_HEIGHT_CM, record_video = True, detect_interval = 1)
#flowsensor = FlowSensor()
record = []

while True:
   sensor_distance, sensor_level = measure_level(REAL_TANK_HEIGHT_CM)
   #flow = flowsensor.measure_flow()
   print('')
   cam_level, cam_distance, status = detect.detect()
   if status == True:
       print(f'level :{sensor_level:.2f}% | height: {sensor_distance:.2f} cm \
            level :{cam_level:.2f}% | height: {cam_distance:.2f} cm')
       record.append([sensor_level, cam_level])
   time.sleep(.1)
   
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cv2.destroyAllWindows()
df = pd.DataFrame(record)
df.to_csv('level record.csv')