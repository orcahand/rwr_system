import os 
import yaml
import re
import numpy as np
import time
from datetime import datetime

# This is a class that contains the calibration functions for the hand_controller. 
# The hand_controller class should inherit from this class and call the functions in this class to calibrate the hand.
class CalibrationClass():

    def __init__(self):
        if type(self) is CalibrationClass:
            raise TypeError("CalibrationClass cannot be instantiated directly. It must be inherited.")
    
    def move_to_desired_positions(self, desired_positions, position_increment=0.2, threshold=0.2):
        """
        Incrementally move motors to their desired positions.

        :param desired_positions: List of desired target positions for each motor.
        :param calibration_current: Current to apply to motors during movement.
        :param position_increment: Maximum step size to increment or decrement the motor position.
        :param threshold: Minimum change in position to consider a motor as having reached its target.
        :return: Final motor positions.
        """
        # Get the initial positions of all motors
        current_positions = self.get_motor_pos()

        # Run up to 4 seconds to reach the desired position. This is done because is oftens get stucκ to a close 
        # enough position it cannot reach.
        start_time = time.time()
        while time.time() - start_time < 4:
            # Compute the difference between desired and current positions
            position_differences = desired_positions - current_positions

            # Calculate step adjustments for each motor
            step_adjustments = np.clip(position_differences, -position_increment, position_increment)

            # Calculate the new target positions
            new_positions = current_positions + step_adjustments

            # Command motors to move to the new target positions
            self.write_desired_motor_pos(new_positions)
            time.sleep(0.08)

            # Update the current positions
            current_positions = self.get_motor_pos()

            # Check if all motors are within the threshold of their desired positions
            if np.all(abs(desired_positions - current_positions) <= threshold):
                break

        return current_positions

    # def move_to_limit_and_get_pos(self, motor_start, calibration_current = 180, position_increment=0.08, threshold=0.0002):
    def move_to_limit_and_get_pos(self, motor_start, calibration_current=180, position_increment=0.1, threshold=0.0002):
        """
        Incrementally move motors to their limits based on motor_start directions.
        Once all active motors (where motor_start != 0) have a position change smaller than the threshold,
        sample all those motors together 10 times and store the mean as the final position.

        :param motor_start: Array of direction values for motors (non-zero to move, 0 to ignore).
        :param calibration_current: Current applied to all motors.
        :param position_increment: Step size to update the motor position.
        :param threshold: Minimum change in position to consider a motor as having reached its limit.
        :return: Array of final, precisely sampled motor positions.
        """
        # Get the initial motor positions and initialize arrays:
        motor_positions = self.get_motor_pos()
        target_positions = motor_positions.copy()
        precise_positions = motor_positions.copy()

        # Apply the calibration current to all motors.
        self.write_desired_motor_current(calibration_current * np.ones(len(self.motor_ids)))
        
        start_time = time.time()
        while True:
            # Update target positions only for motors still moving (motor_start != 0)
            target_positions += position_increment * (motor_start > 0) - position_increment * (motor_start < 0)
            self.write_desired_motor_pos(target_positions)
            time.sleep(0.07)
            
            # Get current positions and compute the change from the previous iteration
            current_positions = self.get_motor_pos()
            position_changes = abs(current_positions - motor_positions)

            # Identify indices of active motors (motor_start != 0)
            active_indices = [i for i, direction in enumerate(motor_start) if direction != 0]

            # Check if all active motors have changes below the threshold
            if active_indices and all(position_changes[i] <= threshold for i in active_indices):
                samples = {i: [] for i in active_indices}
                # Take 10 samples together for all active motors
                for _ in range(4):
                    positions = self.get_motor_pos()
                    for i in active_indices:
                        samples[i].append(positions[i])
                        if i == 5:
                            print("positon of motor {} is {}".format(i, positions[i]))
                    time.sleep(0.05)

                # Compute the mean for each active motor and update the precise_positions array.
                for i in active_indices:
                    precise_positions[i] = np.mean(samples[i])
                    motor_start[i] = 0  # Stop further movement for this motor.

                print("Sampled active motors together. Ids = {}".format(active_indices + np.ones(len(active_indices))))

            motor_positions = current_positions

            if np.all(motor_start == 0):
                print("All targeted motors reached their limits.")
                break

            if time.time() - start_time >= 15:
                print("Time limit reached")
        
        self.momentarily_release_torque()
        return precise_positions

    def auto_calibrate_fingers_with_pos(self, calib_current=120, maxCurrent: int = 150):
            """
            Calibrate each finger by extending the MCP joint fully in both directions and recording the motor positions.
            """
            motors_directions = np.ones(len(self.motor_ids))

            thumb_motor_ids = self.motor_ids_dict["thumb"]
            thumb_motor_idxs = [self.motor_ids.tolist().index(motor_id) for motor_id in thumb_motor_ids]      
            index_motor_ids = self.motor_ids_dict["index"]
            index_motor_idxs = [self.motor_ids.tolist().index(motor_id) for motor_id in index_motor_ids]    
            middle_motor_ids = self.motor_ids_dict["middle"]
            middle_motor_idxs = [self.motor_ids.tolist().index(motor_id) for motor_id in middle_motor_ids]    
            ring_motor_ids = self.motor_ids_dict["ring"]
            ring_motor_idxs = [self.motor_ids.tolist().index(motor_id) for motor_id in ring_motor_ids]    
            pinky_motor_ids = self.motor_ids_dict["pinky"]
            pinky_motor_idxs = [self.motor_ids.tolist().index(motor_id) for motor_id in pinky_motor_ids]    
            
            # Move the index and middle finger ABD in the opposite direction
            motors_directions[middle_motor_idxs[0]] = -1 
            motors_directions[index_motor_idxs[0]] = -1 
            motors_directions[ring_motor_idxs[0]] = -1 
            motors_directions[pinky_motor_idxs[0]] = -1 
            motors_directions[thumb_motor_idxs[0]] = -1 
            motors_directions[thumb_motor_idxs[1]] = -1 

            # current_path = os.path.abspath(__file__)
            # current_path = os.path.dirname(current_path)
            # file_path = os.path.join(current_path,"calibration_yaml", "calibration_ratios.yaml")

            # Create a new calibration file
            date_created = datetime.now().strftime("%Y-%m-%d_%H-%M")
            current_path = os.path.abspath(__file__)
            current_path = os.path.dirname(current_path)
            file_path = os.path.join(current_path, "calibration_"+ date_created+".yaml")        
                
            self.create_yaml_for_calibration([muscle_group.name for muscle_group in self.muscle_groups], file_path)
            
            # Calibrate the wrist pitch joint and keep it's initial position value.
            wrist_init_pos = self.calibrate_wrist_pitch_pos(file_path, 20, maxCurrent, skip_calibration=True)
            # Open the YAML file
            with open(file_path, "r") as yaml_file:
                calibration_defs = yaml.safe_load(yaml_file)
            
            wrist_motor_id = self.motor_ids_dict["wrist"][0]
            wrist_motor_idx = self.motor_ids.tolist().index(wrist_motor_id)

            motor_pos_calib = np.ones(len(self.motor_ids))*calib_current
            motor_pos_calib[wrist_motor_idx] = 0

            self.motor_id2init_pos = self.move_to_limit_and_get_pos(-1*motor_pos_calib*motors_directions, calibration_current=calib_current)
            
            self.momentarily_release_torque()

            # Give fully extended position of wrist found before
            self.motor_id2init_pos[wrist_motor_idx] = wrist_init_pos

            # This is the inital position of the hand. --> This is the .cal file
            self.update_motorinitpos(self.motor_id2init_pos)

            # All motors should be fully extended in one direction at this point
            # TODO: Delete these and there instances to clean up the code
            abd_pos_mean_list = []
            abd_motors_id_map_idx_list = []
            mcp_pos_mean_list = []
            mcp_motors_id_map_idx_list = []
            pip_pos_mean_list = []
            pip_motors_id_map_idx_list = []
            for muscle_group in self.muscle_groups:
                # For the moment we exclude thumb because probably different way to calibrate that
                abd_joint_index = 0  # Assuming ABD joint is the first joint in each muscle group
                mcp_joint_index = 1  # Assuming MCP joint is the second joint in each muscle group
                pip_joint_index = 2  # Assuming PIP joint is the third joint in each muscle group
                if muscle_group.name in ["index", "middle", "ring", "pinky", "thumb"]:

                    if muscle_group.name == "thumb":
                        # MCP is before ABD in thumb
                        abd_joint_index, mcp_joint_index = mcp_joint_index, abd_joint_index
                        dip_joint_index = 3  
                        # DIP
                        # Get the motor id for the DIP joint
                        dip_motor_id = muscle_group.motor_ids[dip_joint_index]
                        # Get the index of the motor id in the motor_ids list
                        dip_motors_id_map_idx = self.motor_ids.tolist().index(dip_motor_id)
                        dip_pos_extended = self.motor_id2init_pos[dip_motors_id_map_idx]

                    # ABD
                    # Get the motor id for the ABD joint
                    abd_motor_id = muscle_group.motor_ids[abd_joint_index]
                    # Get the index of the motor id in the motor_ids list
                    abd_motors_id_map_idx = self.motor_ids.tolist().index(abd_motor_id)
                    abd_motors_id_map_idx_list.append(abd_motors_id_map_idx)

                    # MCP
                    # Get the motor id for the MCP joint
                    mcp_motor_id = muscle_group.motor_ids[mcp_joint_index]
                    # Get the index of the motor id in the motor_ids list
                    mcp_motors_id_map_idx = self.motor_ids.tolist().index(mcp_motor_id)
                    mcp_motors_id_map_idx_list.append(mcp_motors_id_map_idx)

                    # PIP
                    # Get the motor id for the PIP joint
                    pip_motor_id = muscle_group.motor_ids[pip_joint_index]
                    # Get the index of the motor id in the motor_ids list
                    pip_motors_id_map_idx = self.motor_ids.tolist().index(pip_motor_id)
                    pip_motors_id_map_idx_list.append(pip_motors_id_map_idx)

                    # Get extended motor position for both ABD,MCP and PIP joint
                    abd_pos_extended = self.motor_id2init_pos[abd_motors_id_map_idx]
                    mcp_pos_extended = self.motor_id2init_pos[mcp_motors_id_map_idx]
                    pip_pos_extended = self.motor_id2init_pos[pip_motors_id_map_idx]

                    # Move the MCP joint to the opposite direction
                    motor_pos = np.zeros(len(self.motor_ids))
                    motor_pos[mcp_motors_id_map_idx] = -calib_current
                    motor_pos[wrist_motor_idx] = 0

                    mcp_pos_flexed = self.move_to_limit_and_get_pos(-1*motor_pos*motors_directions, calibration_current=calib_current)[mcp_motors_id_map_idx]
                    
                    mcp_pos_mean_list.append(np.mean([mcp_pos_extended, mcp_pos_flexed]))
                    mcp_pos_diff = np.rad2deg(np.abs(mcp_pos_extended - mcp_pos_flexed))

                    mcp_rom_range = muscle_group.joint_roms[mcp_joint_index][1] - muscle_group.joint_roms[mcp_joint_index][0]

                    # Move MCP back to fully extened position. 
                    # Flex the PIP joints and move the ABD joint to the opposite direction.
                    motor_pos[abd_motors_id_map_idx] = -calib_current
                    motor_pos[mcp_motors_id_map_idx] = calib_current
                    motor_pos[pip_motors_id_map_idx] = -calib_current

                    if muscle_group.name == "thumb":
                        motor_pos[dip_motors_id_map_idx] = -calib_current

                    motor_pos[wrist_motor_idx] = 0

                    motor_pos_res = self.move_to_limit_and_get_pos(-1*motor_pos*motors_directions, calibration_current=calib_current)

                    # Get flexed ABD flexed motor position
                    abd_pos_flexed = motor_pos_res[abd_motors_id_map_idx]
                    abd_pos_diff = np.rad2deg(np.abs(abd_pos_extended-abd_pos_flexed))

                    # Get mean value in order to put the findger in the midle after calibration
                    abd_pos_mean_list.append(np.mean([abd_pos_extended, abd_pos_flexed]))
                    abd_rom_range = muscle_group.joint_roms[abd_joint_index][1] - muscle_group.joint_roms[abd_joint_index][0]

                    # Get flexed PIP flexed motor position
                    pip_pos_flexed = motor_pos_res[pip_motors_id_map_idx]
                    pip_pos_diff = np.rad2deg(np.abs(pip_pos_extended-pip_pos_flexed))

                    pip_pos_mean_list.append(np.mean([pip_pos_extended, pip_pos_flexed]))
                    pip_rom_range = muscle_group.joint_roms[pip_joint_index][1] - muscle_group.joint_roms[pip_joint_index][0]

                    # Save the end and start value of the motor position and save the ration
                    calibration_defs[muscle_group.name]["ABD"]["value"] = [float(abd_pos_flexed), float(abd_pos_extended)]
                    # calibration_defs[muscle_group.name]["ABD"]["ratio"] = float(abd_pos_diff/abd_rom_range)
                    calibration_defs[muscle_group.name]["ABD"]["ratio"] = float(np.ceil((abd_pos_diff/abd_rom_range)*10) / 10)

                    
                    # Save the end and start value of the motor position and save the ration
                    calibration_defs[muscle_group.name]["MCP"]["value"] = [float(mcp_pos_flexed), float(mcp_pos_extended)]
                    # calibration_defs[muscle_group.name]["MCP"]["ratio"] = float(mcp_pos_diff/mcp_rom_range)
                    calibration_defs[muscle_group.name]["MCP"]["ratio"] = float(np.ceil((mcp_pos_diff/mcp_rom_range)*10) / 10)


                    # Save the end and start value of the motor position and save the ration
                    calibration_defs[muscle_group.name]["PIP"]["value"] = [float(pip_pos_flexed), float(pip_pos_extended)]
                    # calibration_defs[muscle_group.name]["PIP"]["ratio"] = float(pip_pos_diff/pip_rom_range)
                    calibration_defs[muscle_group.name]["PIP"]["ratio"] = float(np.ceil((pip_pos_diff/pip_rom_range)*10) / 10)

                    if muscle_group.name == "thumb":
                        dip_pos_flexed = motor_pos_res[dip_motors_id_map_idx]
                        dip_pos_diff = np.rad2deg(np.abs(dip_pos_extended-dip_pos_flexed))
                        dip_rom_range = muscle_group.joint_roms[dip_joint_index][1] - muscle_group.joint_roms[dip_joint_index][0]
                        calibration_defs[muscle_group.name]["DIP"]["value"] = [float(dip_pos_flexed), float(dip_pos_extended)]
                        # calibration_defs[muscle_group.name]["DIP"]["ratio"] = float(dip_pos_diff/dip_rom_range)
                        calibration_defs[muscle_group.name]["DIP"]["ratio"] = float(np.ceil((dip_pos_diff/dip_rom_range)*10) / 10)


            # Write the structure to a YAML file
            with open(file_path, "w") as yaml_file:
                yaml.dump(calibration_defs, yaml_file, default_flow_style=False)

            self.set_operating_mode(5)
            self.write_desired_motor_current(maxCurrent * np.ones(len(self.motor_ids)))
            time.sleep(0.2)


    def momentarily_release_torque(self, release_time=0.2):
        """
        Momentarily disables the torque on all Dynamixel motors, allowing them to release 
        their applied torque, then re-enables the torque after a short delay.

        Parameters:
            release_time (float): The duration (in seconds) for which the torque remains disabled.
        """

        # TODO: Remove motor ID of wrist. 
        self.disable_torque(self.motor_ids[:-1])
        
        # Wait for the specified duration to allow the motors to relax.
        time.sleep(release_time)
        
        self.enable_torque(self.motor_ids)


    def calibrate_wrist_pitch_pos(self, yaml_file_path, calib_current=160, maxCurrent: int = 150, skip_calibration=False):
        """
        Calibrate the wrist joint by extending the pitch movement fully in both directions
        and recording the motor positions. Reads and writes results to the provided YAML file.
        
        Args:
            yaml_file_path (str): Path to the existing YAML calibration file.
            calib_current (int): Current value for calibration (default is 40).

        Returns:
            dict: Updated calibration results for the wrist pitch joint.
        """
        # Initialize motor position calibration array
        # Get the motor ID for the wrist pitch joint
        
        wrist_motor_id = self.motor_ids_dict["wrist"][0]
        wrist_motor_idx = self.motor_ids.tolist().index(wrist_motor_id)

        if not skip_calibration:
            motor_pos_calib = np.zeros(len(self.motor_ids))

            # Apply calibration current in one direction and record position
            motor_pos_calib[wrist_motor_idx] = calib_current
            wrist_pos_extended = self.move_to_limit_and_get_pos(motor_pos_calib, calibration_current=calib_current, position_increment=0.055, threshold=0.035)[wrist_motor_idx]

            # Apply calibration current in the opposite direction and record position
            motor_pos_calib[wrist_motor_idx] = -calib_current
            wrist_pos_flexed = self.move_to_limit_and_get_pos(motor_pos_calib, calibration_current=calib_current, position_increment=0.055, threshold=0.035)[wrist_motor_idx]

            # Calculate the range of motion (ROM) for the wrist joint
            wrist_pos_diff = np.rad2deg(np.abs(wrist_pos_extended - wrist_pos_flexed))
            
            wrist_min_rom, wrist_max_rop = self.mano_joints_rom_list[self.mano_joint_mapping["wrist"]][0] 
            wrist_rom_range = wrist_max_rop - wrist_min_rom

            # Calculate the mean position to center the wrist after calibration
            wrist_pos_mean = np.mean([wrist_pos_extended, wrist_pos_flexed])

            # Load the existing calibration definitions
            with open(yaml_file_path, "r") as yaml_file:
                calibration_defs = yaml.safe_load(yaml_file)

            # Save the start and end values, and the ratio for the wrist joint
            calibration_defs["wrist"] = {
                "PITCH": {
                    "value": [float(wrist_pos_flexed), float(wrist_pos_extended)],
                    "ratio": float(wrist_pos_diff / wrist_rom_range)
                }
            }

            # Write the updated calibration definitions back to the YAML file
            with open(yaml_file_path, "w") as yaml_file:
                yaml.dump(calibration_defs, yaml_file, default_flow_style=False)

        else: 
            with open(yaml_file_path, "r") as yaml_file:
                calibration_defs = yaml.safe_load(yaml_file)

            values = calibration_defs['wrist']['PITCH']['value']
            wrist_pos_extended = values[1]
            wrist_pos_mean = sum(values) / len(values)

        motor_pos_wrist = self.get_motor_pos()
        motor_pos_wrist[wrist_motor_idx] = wrist_pos_mean
        self.move_to_desired_positions(motor_pos_wrist)

        # Set the motor position to the mean position for centering the wrist
        time.sleep(0.2)
        return wrist_pos_extended


    def create_yaml_for_calibration(self, finger_names, file_path):
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        try:
            # Get the latest calibration file from the folder
            latest_file = self.find_latest_calibration_file(directory)
            with open(latest_file, "r") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            # Fall back to creating a new structure if no previous file is found
            joints = ["ABD", "MCP", "PIP"]
            data = {}
            for finger in finger_names:
                data[finger] = {}
                joints_to_use = joints + (["DIP"] if finger == "thumb" else (["PITCH"] if finger == "wrist" else []))
                if not joints_to_use:  # if not thumb or wrist, use default joints
                    joints_to_use = joints
                for joint in joints_to_use:
                    data[finger][joint] = {"value": [0, 0], "ratio": 0}

        # Write the (copied or new) data to the new YAML file
        with open(file_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


    def find_latest_calibration_file(self, folder_path):
        # Regex for the naming scheme: calibration_YYYY-MM-DD_HH-MM.yaml
        pattern = r"calibration_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})\.yaml"
        latest_file = None
        latest_time = None

        for file_name in os.listdir(folder_path):
            match = re.match(pattern, file_name)
            if match:
                year, month, day, hour, minute = map(int, match.groups())
                file_time = datetime(year, month, day, hour, minute)
                if latest_time is None or file_time > latest_time:
                    latest_time = file_time
                    latest_file = file_name

        if latest_file is None:
            raise FileNotFoundError("No calibration files found with the specified naming scheme.")

        print("Latest calibration file found:", latest_file)
        return os.path.join(folder_path, latest_file)
    

    # def move_to_limit_and_get_pos(self, motor_start, calibration_current=100, current_increment=10, max_current=200, movement_threshold=0.002, sample_count=4, sample_delay=0.05, timeout=15):
    #     """
    #     Moves motors to their mechanical limits using pure current control with gradual ramping.
        
    #     This function first switches the motors into CURRENT_CONTROL mode (mode 0) and then:
    #       - Commands an initial current (calibration_current) multiplied by motor_start (which indicates the desired direction)
    #       - Monitors the motor positions. For each motor:
    #           • If the position change (delta) between iterations is above the movement_threshold,
    #             we assume the motor is moving freely and do not increase the current.
    #           • If the delta is below the threshold and the commanded current is below max_current,
    #             the current is increased gradually (by current_increment).
    #           • If the motor is already at max_current and the delta remains low, we mark that motor as finished.
    #     Once all targeted motors are finished or the timeout is reached, the function samples the motor positions
    #     several times and returns their average.
        
    #     Parameters:
    #       motor_start (np.array): Array indicating desired movement directions (1 or -1) for each motor (0 to ignore).
    #       calibration_current (int): Starting current (in mA) for all motors.
    #       current_increment (int): Amount to increase the current when movement is insufficient.
    #       max_current (int): Maximum current allowed.
    #       movement_threshold (float): Threshold for detecting significant movement.
    #       sample_count (int): Number of final samples to average.
    #       sample_delay (float): Delay between final samples (in seconds).
    #       timeout (float): Maximum time (in seconds) to run the calibration loop.
        
    #     Returns:
    #       np.array: Final averaged motor positions.
    #     """
    #     # Switch motors to current control mode (mode 0)
    #     self.set_operating_mode(0)
        
    #     previous_positions = self.get_motor_pos()
    #     current_cmd = np.full(len(self.motor_ids), calibration_current, dtype=float)
    #     finished = np.zeros(len(self.motor_ids), dtype=bool)
        
    #     start_time = time.time()
        
    #     # Command the initial current for all motors (use motor_start to set the same direction for all)
    #     self._dxc.write_desired_current(self.motor_ids, current_cmd * motor_start)
        
    #     while time.time() - start_time < timeout:
    #         current_positions = self.get_motor_pos()
    #         delta = np.abs(current_positions - previous_positions)
            
    #         # Check each motor: if movement is small, ramp up current gradually.
    #         for i in range(len(self.motor_ids)):
    #             # Skip motors not commanded or already finished.
    #             if motor_start[i] == 0 or finished[i]:
    #                 continue
    #             if delta[i] >= movement_threshold:
    #                 # Motor is moving sufficiently, so do not change current.
    #                 pass
    #             else:
    #                 # Not moving much: increase current if not already at max.
    #                 if current_cmd[i] < max_current:
    #                     current_cmd[i] = min(current_cmd[i] + current_increment, max_current)
    #                 else:
    #                     # At max and still little movement: assume limit reached.
    #                     finished[i] = True
            
    #         # Apply the updated current commands.
    #         self._dxc.write_desired_current(self.motor_ids, current_cmd * motor_start)
            
    #         previous_positions = current_positions.copy()
            
    #         # Exit when all targeted motors are finished.
    #         if np.all((motor_start == 0) | finished):
    #             print("All targeted motors reached their mechanical limits.")
    #             break
            
    #         time.sleep(0.05)
    #     else:
    #         print("Timeout reached during calibration.")
        
    #     # Once done, sample the final positions several times and average.
    #     samples = np.zeros((sample_count, len(self.motor_ids)))
    #     for s in range(sample_count):
    #         samples[s, :] = self.get_motor_pos()
    #         time.sleep(sample_delay)
    #     final_positions = np.mean(samples, axis=0)
        
    #     self.momentarily_release_torque()
    #     return final_positions
