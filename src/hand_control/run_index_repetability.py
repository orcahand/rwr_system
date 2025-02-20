#!/usr/bin/env python3
import time
import math
import numpy as np
import csv
from hand_controller import HandController

def main():
    # Instantiate the hand controller (adjust the port as needed).
    hc = HandController("/dev/ttyUSB0", calibration=False, auto_calibrate=False)
    time.sleep(1.0)  # Allow time for connection stabilization

    # Retrieve the range of motion (ROM) for each mano joint (in degrees)
    rom_list = hc.get_mano_joints_rom_list()

    # Set all joints to their midpoint to keep them static
    joint_angles = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    # For the index finger (mano mapping "index": slice(5,8)):
    # Joint 5: ABD (fixed)
    # Joint 6: MCP (sinusoidal movement)
    # Joint 7: PIP (sinusoidal movement)
    mcp_rom = rom_list[9]  # MCP joint ROM for index finger
    pip_rom = rom_list[10]  # PIP joint ROM for index finger

    # Calculate midpoints and amplitudes for sinusoidal motion (in degrees)
    mcp_mid = (mcp_rom[0] + mcp_rom[1]) / 2.0
    mcp_amp = (mcp_rom[1] - mcp_rom[0]) / 2.0
    pip_mid = (pip_rom[0] + pip_rom[1]) / 2.0
    pip_amp = (pip_rom[1] - pip_rom[0]) / 2.0

    # Open a CSV file for logging the commanded joint angles.
    # Each row will contain a timestamp followed by all 17 joint angles.
    csv_filename = "angles_log.csv"
    csv_file = open(csv_filename, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["timestamp"] + [f"joint_{i}" for i in range(len(joint_angles))]
    csv_writer.writerow(header)

    # Record the time when calibration was last performed.
    last_calibration_time = time.time()

    print("Starting sinusoidal movement for index MCP and PIP joints.")
    try:
        counter = 0
        while True:
            current_time = time.time()

            # After 1 minute, run the self-calibration routine.
            # if current_time - last_calibration_time >= 60:
            #     print("1 minute elapsed. Running self-calibration...")
            #     hc.auto_calibrate_fingers_with_pos(calib_current=120, maxCurrent=150)
            #     print("Self-calibration complete. Resuming sinusoidal motion.")
            #     last_calibration_time = current_time

            # Compute the phase using the interval counter (200 steps per full sine wave)
            phase = (counter / 800.0) * 2 * math.pi
            joint_angles[9] = mcp_mid + mcp_amp * math.sin(phase)
            joint_angles[10] = pip_mid + pip_amp * math.sin(phase)

            print("MCP is {}".format(joint_angles[9]))
            print("PIP is {}".format(joint_angles[10]))

            # Log the current commanded joint angles to CSV (with timestamp).
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
            csv_writer.writerow([timestamp] + list(joint_angles))
            csv_file.flush()

            # Convert the desired joint angles (in degrees) to motor positions (in radians)
            hc.write_desired_joint_angles(joint_angles)

            # Increment counter and wrap it around every 200 steps
            counter = (counter + 1) % 800

            time.sleep(0.02)  # Update at approximately 20Hz


    except KeyboardInterrupt:
        print("Sinusoidal motion interrupted. Terminating hand controller.")
        hc.terminate()
        csv_file.close()

if __name__ == "__main__":
    main()
