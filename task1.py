import pigpio
import time

pi = pigpio

LED_PIN = 12
pi.set_mode(LED_PIN, pigpio.OUTPUT)

while True:
    pi.write(LED_PIN, 1)
    time.sleep(1)
    pi.write(LED_PIN, 0)
    time.sleep(1)

pi.stop()