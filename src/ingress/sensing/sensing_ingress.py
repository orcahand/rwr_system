#!/usr/bin/env python3

import random
import time
import threading
import serial
import json

class ArduinoDriver:
    def __init__(self, callback, baud_rate=9600, device=None, debug=False):
        self.callback = callback
        self.baud_rate = baud_rate
        self.device = device
        self.running = False
        self.serial_connection = None
        self.debug = debug

    def run(self):
        self.running = True
        if self.device:
            self.serial_connection = serial.Serial(self.device, self.baud_rate)
            self.read_from_device()
        else:
            # Generate random values if no device is available
            self.generate_random_values()

    def stop(self):
        self.running = False
        if self.serial_connection:
            self.serial_connection.close()


    def read_from_device(self):
        while self.running:
            try:
                line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    # Convert space-separated values to a list of floats
                    values = list(map(float, line.split()))
                    # Send as dictionary (modify keys as needed)
                    data = {
                        "thumb": values[2],
                        "index": values[1],
                        "middle": values[3],
                        "ring": values[0],
                        "pinky": values[4]
                    }
                    self.callback(data)
            except ValueError:
                print(f"Invalid numeric data received: {line}")
            except Exception as e:
                print(f"Error reading from device: {e}")
            time.sleep(0.01)


    def generate_random_values(self):
        while self.running:
            # Generate random sensor data
            data = {
                "pressure": {
                    "thumb": random.uniform(0, 1),
                    "index": random.uniform(0, 1),
                    "middle": random.uniform(0, 1),
                    "ring": random.uniform(0, 1),
                    "pinky": random.uniform(0, 1)
                },
                "fsr": {
                    "thumb": random.uniform(0, 1),
                    "index": random.uniform(0, 1),
                    "middle": random.uniform(0, 1),
                    "ring": random.uniform(0, 1),
                    "pinky": random.uniform(0, 1)
                }
            }
            self.callback(data)
            time.sleep(0.1)

if __name__ == "__main__":
    def example_callback(data):
        print(f"Received data: {data}")

    device = "/dev/serial/by-id/usb-Arduino_LLC_Arduino_Nano_Every_8CB3769751544E5450202020FF054012-if00"
    baud_rate = 9600

    driver = ArduinoDriver(callback=example_callback, device=device, baud_rate=baud_rate, debug=True)
    driver_thread = threading.Thread(target=driver.run)
    driver_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        driver.stop()
        driver_thread.join()