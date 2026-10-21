import RPi.GPIO as GPIO
import time

class FlowSensor():
    def __init__(self, bouncetime = 5):
        self.pin = 22
        self.bouncetime = bouncetime
        self.pulse_count = 0
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.pin,
                              GPIO.FALLING,
                              callback=self._count_pulse,
                              bouncetime = self.bouncetime)
        
    def _count_pulse(self, channel):
        self.pulse_count += 1
    
    def _reset_count(self):
        self.pulse_count = 0
    
    def measure_flow(self):
        self._reset_count()
        start_time = time.time()
        time.sleep(1)
        count = self.pulse_count
        elapsed = time.time() - start_time
        
        freq = count / elapsed
        flowrate = freq / 7.5
        
        print(f'pulse: {self.pulse_count} | flowrate: {flowrate:.2f} L/min')
        return flowrate

    
    
