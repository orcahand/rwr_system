#!/usr/bin/env python3
import time
import math
import numpy as np
import csv
from hand_controller import HandController

# Select finger
FINGER = "middle"

def get_finger_joints(finger):
    mapping = {
        "index": (6, 7),
        "middle": (9, 10),
        "ring": (12, 13),
        "pinky": (15, 16)
    }
    return mapping.get(finger, (6, 7))

def main():
    hc = HandController("/dev/ttyUSB0", calibration=False, auto_calibrate=False)
    time.sleep(1.0)

    rom_list = hc.get_mano_joints_rom_list()
    joint_angles = np.zeros(17)

    mcp_idx, pip_idx = get_finger_joints(FINGER)
    mcp_rom = rom_list[mcp_idx]
    pip_rom = rom_list[pip_idx]

    mcp_mid = (mcp_rom[0] + mcp_rom[1]) / 2.0
    mcp_amp = (mcp_rom[1] - mcp_rom[0]) / 2.0
    pip_mid = (pip_rom[0] + pip_rom[1]) / 2.0
    pip_amp = (pip_rom[1] - pip_rom[0]) / 2.0

    # Prepare CSV
    csv_file = open("angles_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp"] + [f"joint_{i}" for i in range(len(joint_angles))])

    print(f"Starting sinusoidal movement for {FINGER} finger MCP and PIP joints.")

    mcp_counter = 0
    pip_counter = 0
    mcp_reversing = False  # To track if we've already flipped this wave cycle

    try:
        while True:
            now = time.time()
            phase_mcp = (mcp_counter / 800.0) * 2 * math.pi
            phase_pip = (pip_counter / 400.0) * 2 * math.pi

            # Proposed angles
            mcp_angle_proposed = mcp_mid + mcp_amp * math.sin(phase_mcp)
            pip_angle = pip_mid + pip_amp * math.sin(phase_pip)
            sum_angles = mcp_angle_proposed + pip_angle

            # If sum > 180 and we haven't flipped yet, jump to descending portion
            if sum_angles > 180 and not mcp_reversing:
                # Identify the next peak (pi/2 or 3*pi/2) to skip over
                if phase_mcp < math.pi/2:
                    next_peak = math.pi/2
                else:
                    next_peak = 3*math.pi/2

                diff = next_peak - phase_mcp
                phase_mcp += 2 * diff  # jump past the peak
                phase_mcp %= 2 * math.pi  # wrap within [0, 2π)
                mcp_counter = int((phase_mcp / (2 * math.pi)) * 800)

                # Recompute after jump
                mcp_angle_proposed = mcp_mid + mcp_amp * math.sin(phase_mcp)
                mcp_reversing = True  # we are now in "descending" mode

            # If we already flipped and angles are comfortably below 180, allow next flip
            if mcp_reversing and sum_angles < 160:
                mcp_reversing = False

            # Set final angles
            joint_angles[mcp_idx] = mcp_angle_proposed
            joint_angles[pip_idx] = pip_angle

            print(f"{FINGER.capitalize()} MCP: {mcp_angle_proposed:.2f}")
            print(f"{FINGER.capitalize()} PIP: {pip_angle:.2f}")

            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
            csv_writer.writerow([stamp] + list(joint_angles))
            csv_file.flush()
            hc.write_desired_joint_angles(joint_angles)

            mcp_counter = (mcp_counter + 1) % 800
            pip_counter = (pip_counter + 1) % 400
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Interrupted. Terminating.")
    finally:
        hc.terminate()
        csv_file.close()

if __name__ == "__main__":
    main()
