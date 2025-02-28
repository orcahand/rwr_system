import click
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import os


def find_stable_windows_reverse(array, threshold, min_length):
    windows = []
    window_end = len(array) - 1
    for window_start in range(len(array) - 1, -1, -1):
        variation = max(array[window_start:window_end+1]) - min(array[window_start:window_end+1])
        if variation > threshold:
            if window_end - window_start >= min_length:
                windows.append((window_start+1, window_end))  # Adjusted window_start and window_end accordingly
            window_end = window_start + 1  # Adjusted window_end accordingly
    # if window_end - window_start >= min_length:
    #     windows.append((window_start+1, window_end))
    return windows[::-1]  # Reverse the order of windows before returning

def calc_latency(windows_gt, gt_time, windows_cmd, cmd_angles, cmd_time, lower_bound, higher_bound, axes, axis_index, latency_distance):

    low_up_latency = []
    low_down_latency = []
    mid_up_latency = []
    mid_down_latency = []
    high_up_latency = []
    high_down_latency = []

    command_state = -1
    for gt_w in windows_gt:
        target_value = gt_time[gt_w[0]]
        
        # Iterate through windows_cmd
        for cmd_w in windows_cmd:
            cmd_value = cmd_time[cmd_w[0]]
            
            if cmd_value < target_value and (target_value - cmd_value) < latency_distance:
                command_height = cmd_angles[cmd_w[0]]

                if command_height != 0:
                    axes[axis_index].axvline(target_value, color='b', linestyle='--')
                    axes[axis_index].axvline(cmd_value, color='r', linestyle='--')


                if 0 < command_height < lower_bound:
                    low_up_latency.append(target_value-cmd_value)
                    command_state = 0
               

                elif lower_bound < command_height < higher_bound:
                    mid_up_latency.append(target_value-cmd_value)
                    command_state = 1 


                elif command_height > higher_bound:
                    high_up_latency.append(target_value-cmd_value)
                    command_state = 2


                elif command_height == 0 and command_height < cmd_angles[cmd_w[0]-1]:
                    if command_state != -1:
                        axes[axis_index].axvline(target_value, color='b', linestyle='--')
                        axes[axis_index].axvline(cmd_value, color='r', linestyle='--')

                    if command_state == 0:
                        low_down_latency.append(target_value-cmd_value)
                        
                    if command_state == 1:
                        mid_down_latency.append(target_value-cmd_value)
                        
                    if command_state == 2:
                        high_down_latency.append(target_value-cmd_value)
                                   
    print(f"LOW UP average latency ({len(low_up_latency)} entries): {np.round(np.mean(low_up_latency),4)} +- {np.round(np.std(low_up_latency),4)}")
    print(f"LOW DOWN average latency: ({len(low_down_latency)} entries): {np.round(np.mean(low_down_latency),4)} +- {np.round(np.std(low_down_latency),4)}")
    print(f"MID UP average latency: ({len(mid_up_latency)} entries): {np.round(np.mean(mid_up_latency),4)} +- {np.round(np.std(mid_up_latency),4)}")
    print(f"MID DOWN average latency: ({len(mid_down_latency)} entries): {np.round(np.mean(mid_down_latency),4)} +- {np.round(np.std(mid_down_latency),4)}")
    print(f"HIGH UP average latency: ({len(high_up_latency)} entries): {np.round(np.mean(high_up_latency),4)} +- {np.round(np.std(high_up_latency),4)}")
    print(f"HIGH DOWN average latency: ({len(high_down_latency)} entries): {np.round(np.mean(high_down_latency),4)} +- {np.round(np.std(high_down_latency),4)}")
    return 0



def calc_angle(x_coordinates, y_coordinates, z_coordinates):

    proximal_angles = []
    distal_angles = []

    for i in range(len(x_coordinates[0])):

        
        align_vec1_projected = np.array([x_coordinates[0][i], y_coordinates[0][i], z_coordinates[0][i]])
        align_vec2_projected = np.array([x_coordinates[1][i], y_coordinates[1][i], z_coordinates[1][i]])
        align_vec3_projected = np.array([x_coordinates[2][i], y_coordinates[2][i], z_coordinates[2][i]])

        # Normalize the projected alignment vectors
        align_vec1_projected[:2] /= np.linalg.norm(align_vec1_projected[:2])
        align_vec2_projected[:2] /= np.linalg.norm(align_vec2_projected[:2])
        align_vec3_projected[:2] /= np.linalg.norm(align_vec3_projected[:2])

        # Calculate the dot product between the y-alignment vectors in the z-plane
        dot_product_proximal = np.dot(align_vec1_projected[:2], align_vec2_projected[:2])
        dot_product_distal = np.dot(align_vec2_projected[:2], align_vec3_projected[:2])


        # Calculate the angle in radians
        angle_proximal = math.degrees(math.acos(dot_product_proximal))
        angle_distal = math.degrees(math.acos(dot_product_distal))


        # Append the angles to the respective lists
        proximal_angles.append(angle_proximal)
        distal_angles.append(angle_distal)


    # Convert the angle lists to pandas Series
    proximal_angles = pd.Series(proximal_angles)
    distal_angles = pd.Series(distal_angles)

    return proximal_angles, distal_angles

def moving_avg(coords, window_size):
    coords_ma = coords.rolling(window_size, min_periods=1, center=True).mean()
    coords_ma = coords_ma.tolist()
    return coords_ma


def moving_avg_exponential(coords, exponential_moving_average):
    coords_ema = coords.ewm(span=exponential_moving_average).mean()
    coords_ema = coords_ema.tolist()
    return coords_ema

def moving_avg_gaussian(data, window_size, std_dev):
    weights = np.exp(-(np.arange(window_size) - window_size // 2) ** 2 / (2 * std_dev ** 2))
    weights /= np.sum(weights)
    
    # Pad the data to handle boundary elements
    padding = window_size // 2
    padded_data = np.pad(data, (padding, padding), mode='edge')
    
    moving_avg = np.convolve(padded_data, weights, mode='valid')

    return pd.Series(moving_avg).to_list()


def validate_window_size(ctx, param, value):
    if value is None:
        return None  # No validation needed if the value is None

    if value % 2 != 1:
        raise click.BadParameter("Window Size must be an odd number.")
    return value

@click.command()
@click.option(
    "--moving_average",
    "-ma",
    type=int,
    default=None,
    callback=validate_window_size,
    help="Define MA Window Size",
)
@click.option(
    "--gaussian_moving_average",
    "-g",
    type=float,
    default=None,
    help="Define Standard Deviation of GMA",
)
@click.option(
    "--exponential_moving_average",
    "-e",
    type=int,
    default=None,
    help="Define Decay Factor of EMA",
)

def estimate_joint_angles(moving_average: int, gaussian_moving_average: int, exponential_moving_average: int):

  
    command_df = pd.read_csv("data_files/commanded_joint_states.csv")


    command_timestamps = command_df["Timestamp"]* 1e-9
    command_timestamps = command_timestamps - command_timestamps.iloc[0]
    command_root2index_pp = np.degrees(command_df["root2index_pp"])
    command_index_pp2mp = np.degrees(command_df["index_pp2mp"])
    

    
    data = pd.read_csv('data_files/groundtruth_marker_positions.csv', delimiter=',')

    window_size = moving_average

    column_headers = data.columns.tolist()
    num_markers = len(column_headers) // 3
    window_size = moving_average

    x_coordinates = []
    y_coordinates = []
    z_coordinates = []
    x_coordinates_ma = []
    y_coordinates_ma = []
    z_coordinates_ma = []
    x_coordinates_gma = []
    y_coordinates_gma = []
    z_coordinates_gma = []
    x_coordinates_ema = []
    y_coordinates_ema = []
    z_coordinates_ema = []

    for i in range(num_markers):
        x_col = f"Tag Alignment Vector {i+1}: X-Coord"
        y_col = f"Tag Alignment Vector {i+1}: Y-Coord"
        z_col = f"Tag Alignment Vector {i+1}: Z-Coord"

        x_coords = data[x_col]
        y_coords = data[y_col]
        z_coords = data[z_col]

        x_coordinates.append(x_coords)
        y_coordinates.append(y_coords)
        z_coordinates.append(z_coords)


        if moving_average is not None:
            # Apply moving average filter on coordinates
            x_coords_ma = moving_avg(x_coords, window_size)
            y_coords_ma = moving_avg(y_coords, window_size)
            z_coords_ma = moving_avg(z_coords, window_size)
           
            x_coordinates_ma.append(x_coords_ma)
            y_coordinates_ma.append(y_coords_ma)
            z_coordinates_ma.append(z_coords_ma)


        if gaussian_moving_average is not None:
            if window_size <= 1:
                raise ValueError("Window size must be greater than 1 for polynomial moving average: Include -ma i, with i > 1")

            standard_deviation = gaussian_moving_average
            x_coords_gma = moving_avg_gaussian(x_coords, window_size, standard_deviation)
            y_coords_gma = moving_avg_gaussian(y_coords, window_size, standard_deviation)
            z_coords_gma = moving_avg_gaussian(z_coords, window_size, standard_deviation)

            x_coordinates_gma.append(x_coords_gma)
            y_coordinates_gma.append(y_coords_gma)
            z_coordinates_gma.append(z_coords_gma)
         

        if exponential_moving_average is not None:
            x_coords_ema = moving_avg_exponential(x_coords, exponential_moving_average)
            y_coords_ema = moving_avg_exponential(y_coords, exponential_moving_average)
            z_coords_ema = moving_avg_exponential(z_coords, exponential_moving_average)

            x_coordinates_ema.append(x_coords_ema)
            y_coordinates_ema.append(y_coords_ema)
            z_coordinates_ema.append(z_coords_ema)



    proximal_angles, distal_angles = calc_angle(x_coordinates, y_coordinates, z_coordinates)
    # Print the calculated angle
    

    t_col = data['Timestamp'] * 1e-9
    t_col = t_col - t_col.iloc[0]


    if moving_average is not None:
        proximal_angles_ma, distal_angles_ma = calc_angle(x_coordinates_ma, y_coordinates_ma, z_coordinates_ma)
    if gaussian_moving_average is not None:
        proximal_angles_gma, distal_angles_gma = calc_angle(x_coordinates_gma,y_coordinates_gma,z_coordinates_gma)
    if exponential_moving_average is not None:
        proximal_angles_ema, distal_angles_ema = calc_angle(x_coordinates_ema,y_coordinates_ema,z_coordinates_ema)

    # Create a figure and subplots
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), num=f'Faive Pinky Joint Angles')


    threshold = 1.8
    min_length = 20
    latency_distance = 0.6
    lower_bound = 20
    higher_bound = 70

    windows_gt_prox = find_stable_windows_reverse(proximal_angles, threshold, min_length)
    windows_gt_dist = find_stable_windows_reverse(distal_angles, threshold, min_length)
    windows_cmd_prox = find_stable_windows_reverse(command_root2index_pp, 0, 3)
    windows_cmd_dist = find_stable_windows_reverse(command_index_pp2mp, 0, 3)


    print("PROXIMAL ANGLE LATENCY:")
    calc_latency(windows_gt_prox, t_col, windows_cmd_prox, command_root2index_pp, command_timestamps, lower_bound, higher_bound, axes, 0, latency_distance)
    print("DISTAL ANGLE LATENCY:")
    calc_latency(windows_gt_dist, t_col, windows_cmd_dist, command_index_pp2mp, command_timestamps, lower_bound, higher_bound, axes, 1, latency_distance)

   
 
    axes[0].plot(t_col, proximal_angles, label="Ground Truth")
    #axes[0].plot(joint_timestamps, joint_root2index_pp, label="Proprioception")
    axes[0].plot(command_timestamps, command_root2index_pp, label="Command")

    axes[1].plot(t_col, distal_angles, label="Ground Truth")
    #axes[1].plot(joint_timestamps, joint_index_pp2mp, label="Proprioception")
    axes[1].plot(command_timestamps, command_index_pp2mp, label="Command")
    # Plot the proximal angle
    if moving_average is not None and gaussian_moving_average is None and exponential_moving_average is None:
        axes[0].plot(t_col, proximal_angles_ma,label="Ground Truth [MA]")
        axes[1].plot(t_col, distal_angles_ma,label="Ground Truth [MA]")

    elif gaussian_moving_average is not None and exponential_moving_average is None:
        axes[0].plot(t_col, proximal_angles_gma,label="Ground Truth [GMA]")
        axes[1].plot(t_col, distal_angles_gma,label="Ground Truth [GMA]")

    elif exponential_moving_average is not None:
        axes[1].plot(t_col, distal_angles_ema,label="Ground Truth [EMA]")
        axes[0].plot(t_col, proximal_angles_ema,label="Ground Truth [EMA]")


    axes[0].set_xlabel('Seconds')
    axes[0].set_ylabel('Angle [Degrees]')
    axes[0].set_title(f'Proximal Angle')
    axes[0].legend()
    axes[1].set_xlabel('Seconds')
    axes[1].set_ylabel('Angle [Degrees]')
    axes[1].set_title(f'Distal Angle')
    axes[1].legend()

    # Adjust spacing between subplots
    fig.subplots_adjust(hspace=0.4)

    # latency_data = {'LOW UP': low_up_latency,'LOW DOWN': low_down_latency,'MID UP': mid_up_latency,'MID DOWN': mid_down_latency,'HIGH UP': high_up_latency,'HIGH DOWN': high_down_latency}

  
    # # Determine the maximum length among the arrays
    # max_length = max(len(low_up_latency), len(low_down_latency), len(mid_up_latency), len(mid_down_latency), len(high_up_latency), len(high_down_latency))
    # for key in latency_data:
    #     latency_data[key] += [float('nan')] * (max_length - len(latency_data[key]))

    # latency_df = pd.DataFrame(latency_data)
    # latency_df.to_csv('data_files/latency.csv', index=False)
     
    angle_df = pd.DataFrame({'Timestamp': data['Timestamp'], 'index_pp2mp': proximal_angles, 'index_mp2dp': distal_angles})
    angle_df.to_csv('data_files/groundtruth_joint_states.csv', index=False)


    # Show the plot
    plt.show()

        

if __name__ == "__main__":
    estimate_joint_angles()
