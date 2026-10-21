import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

TRIG = 2
ECHO = 3
calibration = 2.3

GPIO.setup(TRIG ,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)
GPIO.output(TRIG, False)
pulse_start = 0

def measure_level(tank_height, timeout=0.03):
   # Make sure trigger starts LOW
   GPIO.output(TRIG, False)
   time.sleep(0.00005)

   # Send 10 us trigger pulse
   GPIO.output(TRIG, True)
   time.sleep(0.00001)
   GPIO.output(TRIG, False)

   start_wait = time.monotonic()

   # Wait for ECHO to go HIGH
   while GPIO.input(ECHO) == 0:
       if time.monotonic() - start_wait > timeout:
           print("Timeout: ECHO never went HIGH")
           return None, None

   pulse_start = time.monotonic()

   # Wait for ECHO to go LOW
   while GPIO.input(ECHO) == 1:
       if time.monotonic() - pulse_start > timeout:
           print("Timeout: ECHO stayed HIGH too long")
           return None, None

   pulse_stop = time.monotonic()

   pulse_time = pulse_stop - pulse_start

   distance = pulse_time * 17165		# sound speed 343 m/s
   distance += calibration
   
   level = max((1 - distance / tank_height) * 100, 0.0)
   print(f'Distance: {round(distance, 2)} cm')
   print(f'Level: {round(level, 2)}%')
   return distance, level		# return distance in cm