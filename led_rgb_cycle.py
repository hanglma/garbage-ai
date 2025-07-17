import RPi.GPIO as GPIO
import time

# GPIO pin numbers for RGB
RED_PIN = 17
GREEN_PIN = 27
BLUE_PIN = 22

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(BLUE_PIN, GPIO.OUT)

try:
    while True:
        # Red
        GPIO.output(RED_PIN, GPIO.HIGH)
        GPIO.output(GREEN_PIN, GPIO.LOW)
        GPIO.output(BLUE_PIN, GPIO.LOW)
        time.sleep(5)
        # Green
        GPIO.output(RED_PIN, GPIO.LOW)
        GPIO.output(GREEN_PIN, GPIO.HIGH)
        GPIO.output(BLUE_PIN, GPIO.LOW)
        time.sleep(5)
        # Blue
        GPIO.output(RED_PIN, GPIO.LOW)
        GPIO.output(GREEN_PIN, GPIO.LOW)
        GPIO.output(BLUE_PIN, GPIO.HIGH)
        time.sleep(5)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
    print('GPIO cleaned up.')
