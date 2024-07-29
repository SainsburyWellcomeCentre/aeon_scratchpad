import argparse
import os
import re

import h5py
import pandas as pd
import numpy as np

from aeon.io import video
from aeon.io import api
from aeon.schema.schemas import social02

# Globals
FPS = 50
CM_TO_PX = 5.4  # 1 cm = 5.4 px


def process_files(filepaths, export_directory):
    """
    Process each HDF5 file to extract tracks, compute distances, identify chases, and export results.

    Args:
        filepaths (list of str): List of file paths to process.
        export_directory (str): Directory to export the results.
    """
    
    for filepath in filepaths:
        print(f'Processing file: {filepath}')
        
        # Ensure the file exists
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        ## Funtions that do the processing:
        # Extract tracks
        tracks, track_names, video_path = extract_tracks(filepath)
        
        # Comopute distances, orientations, etc
        centroid_distances, relative_distances, extremity_distances, orientations = compute_distances_and_orientations(tracks)
        
        # Extract possible chasing frames
        possible_chasing_frames = find_possible_chasing_frames(centroid_distances, relative_distances, orientations)
        
        # Extract possible chasing events and further restrict based on speed, area, etc
        filtered_chases = get_and_filter_events(possible_chasing_frames, tracks, video_path)
        
        # Determine chaser
        chases, chaser_ids = identify_chaser(filtered_chases, track_names, extremity_distances)
        
        # Export results and further restrict length of events
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)
        export_results(chases, chaser_ids, video_path, export_directory)
        
        print(f"Completed processing for file: {filepath}\n")        


def extract_tracks(filepath):
    # Open the HDF5 file in read mode
    with h5py.File(filepath, 'r') as f:
        # Extract the tracks
        tracks = f['tracks'][:]
        track_names = f['track_names'][:]
        track_names = [name.decode('utf-8') for name in track_names]
        video_path = f['video_path'][()].decode('utf-8')

    return tracks, track_names, video_path


def compute_distances_and_orientations(tracks):
    # Indices for keypoints
    nose_index = 0  
    head_index = 1 
    spine1_index = 4
    spine2_index = 5
    spine3_index = 6 
    spine4_index = 7  

    # Centroid distances 
    centroid_mouse0 = tracks[0, :, spine2_index, :]
    centroid_mouse1 = tracks[1, :, spine2_index, :]
    centroid_distances = np.linalg.norm(centroid_mouse0 - centroid_mouse1, axis=0)

    # Relative spine distances
    spine4_mouse0 = tracks[0, :, spine4_index, :]
    head_mouse0 = tracks[0, :, head_index, :]
    head_mouse1 = tracks[1, :, head_index, :]
    relative_distances = np.zeros((2, tracks.shape[3]))
    relative_distances[0, :] = np.linalg.norm(spine4_mouse0 - head_mouse0, axis=0)
    relative_distances[1, :] = np.linalg.norm(spine4_mouse0 - head_mouse1, axis=0)

    # Extremity distances
    spine4_mouse1 = tracks[1, :, spine4_index, :]
    extremity_distances = np.zeros((4, tracks.shape[3]))
    extremity_distances[0, :] = np.linalg.norm(head_mouse0 - head_mouse1, axis=0)
    extremity_distances[1, :] = np.linalg.norm(spine4_mouse0 - spine4_mouse1, axis=0)
    extremity_distances[2, :] = np.linalg.norm(spine4_mouse0 - head_mouse1, axis=0)
    extremity_distances[3, :] = np.linalg.norm(spine4_mouse1 - head_mouse0, axis=0)

    # Orientation
    # Calculate differences in x and y coordinates
    dy_tail_nose = tracks[:, 1, nose_index, :] - tracks[:, 1, spine4_index, :]
    dx_tail_nose = tracks[:, 0, nose_index, :] - tracks[:, 0, spine4_index, :]
    dy_tail_head = tracks[:, 1, head_index, :] - tracks[:, 1, spine4_index, :]
    dx_tail_head = tracks[:, 0, head_index, :] - tracks[:, 0, spine4_index, :]
    # Calculate angles: 0 degrees if the mice are facing towards the nest, angles increase counterclockwise
    angles_tail_nose = np.degrees(np.arctan2(-dy_tail_nose, dx_tail_nose))
    angles_tail_head = np.degrees(np.arctan2(-dy_tail_head, dx_tail_head))
    # Adjust angles to be in the range [0, 360)
    angles_tail_nose = np.where(angles_tail_nose < 0, angles_tail_nose + 360, angles_tail_nose)
    angles_tail_head = np.where(angles_tail_head < 0, angles_tail_head + 360, angles_tail_head)
    # When angles_tail_nose is NaN, use angles_tail_head
    orientations = np.where(np.isnan(angles_tail_nose), angles_tail_head, angles_tail_nose)

    return centroid_distances, relative_distances, extremity_distances, orientations


def find_possible_chasing_frames(centroid_distances, relative_distances, orientations):
    angle_tolerance = 45
    max_distance = 400

    # Adjust the orientation of mouse 2
    adjusted_orientations = (orientations[1]) % 360
    # this just did modulus operation on the orientation of the second mouse, so that it is in the same range as the first mouse - do we even need this?

    # Condition 1: the mice have opposite orientations, within a certain tolerance
    orientation_condition = np.isclose(orientations[0], adjusted_orientations, atol=angle_tolerance)

    # Condition 2: the distance between the mice's centroids is less than a certain threshold, ensuring they are close to each other
    distance_condition = centroid_distances < max_distance

    # Condition 3: relative spine measure, removes cases where mice are side by side
    relative_distance_condition = relative_distances[1] > relative_distances[0]

    # Find frames where all conditions are true
    possible_chasing_frames = np.where(np.logical_and.reduce([orientation_condition, distance_condition, relative_distance_condition]))[0]

    return possible_chasing_frames


def get_and_filter_events(possible_chasing_frames, tracks, video_path):
    max_frame_gap = 20
    min_possible_chasing_frames = 15

    # Divide possible_chasing_frames into sub_arrays of consecutive frames (allowing for gaps up to a certain max)
    diffs = np.diff(possible_chasing_frames)
    indices = np.where(diffs > max_frame_gap)[0]
    indices += 1
    possible_chasing_events = np.split(possible_chasing_frames, indices)

    # Filter sub_arrays to keep only those with more than a certain number of frames
    possible_chasing_events = [sub_array for sub_array in possible_chasing_events if len(sub_array) > min_possible_chasing_frames]


    # Use regex to match the pattern for the root and the two timestamps
    metadata_retrieval_matches = re.search(r'(.*?)(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}).*(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', video_path)
    arena_number_match = re.search(r'AEON(\d)', video_path)
    # Extract the root part of the path and the two timestamps
    root = metadata_retrieval_matches.group(1)
    start_time = pd.to_datetime(metadata_retrieval_matches.group(2), format='%Y-%m-%dT%H-%M-%S')
    chunk_time = pd.to_datetime(metadata_retrieval_matches.group(3), format='%Y-%m-%dT%H-%M-%S')
    arena_number = arena_number_match.group(1)
    # # Option to change the drive
    new_drive = "/ceph/aeon"
    root = re.sub(r'^.*?:', new_drive, root)

    # Retrieve the metadata
    metadata = api.load(
        root, social02.Metadata, start=start_time, end=chunk_time
    )["metadata"].iloc[0]

    inner_radius = float(metadata.ActiveRegion.ArenaInnerRadius)
    outer_radius = float(metadata.ActiveRegion.ArenaOuterRadius)
    center_x = float(metadata.ActiveRegion.ArenaCenter.X)
    center_y = float(metadata.ActiveRegion.ArenaCenter.Y)
    nest_y1 = float(metadata.ActiveRegion.NestRegion.ArrayOfPoint[1].Y)
    nest_y2 = float(metadata.ActiveRegion.NestRegion.ArrayOfPoint[2].Y)
    if arena_number == '3':
        entrance_y1 = 547
        entrance_y2 = 573
    elif arena_number == '4':
        entrance_y1 = 554
        entrance_y2 = 587

    # Create an array of frame numbers
    frame_numbers = np.arange(tracks.shape[3])

    # Get the x and y coordinates of spine2 for both mice
    spine2_index = 5
    spine2_x = tracks[:, 0, spine2_index, :]
    spine2_y = tracks[:, 1, spine2_index, :]

    # Calculate the squared distance from the center of the ROI
    dist_squared = (spine2_x - center_x)**2 + (spine2_y - center_y)**2


    # Condition 4: both mice in same part of arena

    # Condition 4a: both mice in corridor
    # Check if the distance is within the squared radii for both mice
    within_corridor = (inner_radius**2 <= dist_squared) & (dist_squared <= outer_radius**2)

    # Condition 4b: both mice in cenral part of arena
    # Check if the mice are in the central region (within the inner radius)
    within_center = (inner_radius**2 >= dist_squared)

    #TODO: do we want ot exclue nest or entrance?
    # Check if the mice are in excluded regions
    in_excluded_region_nest = (spine2_x > center_x) & ((spine2_y >= nest_y1) & (spine2_y <= nest_y2))
    in_excluded_region_entrance = (spine2_x < center_x) & ((spine2_y >= entrance_y1) & (spine2_y <= entrance_y2))

    # Update the conditions to exclude the specified regions
    valid_corridor = within_corridor & ~np.any(in_excluded_region_nest | in_excluded_region_entrance, axis=0)
    valid_center = within_center & ~np.any(in_excluded_region_nest | in_excluded_region_entrance, axis=0)

    # Combine conditions: both mice in the corridor or both mice in the center
    both_in_corridor = np.all(valid_corridor, axis=0)
    both_in_center = np.all(valid_center, axis=0)

    # Combine conditions
    both_in_same_area = both_in_corridor | both_in_center

    # Filter the frame numbers where both mice are in the same area
    frame_numbers_in_same_area = frame_numbers[both_in_same_area]

    # Iterate over each event (array of frames)
    filtered_chasing_events = []
    for event in possible_chasing_events:
        # Get the frames where both mice are in the same area for the event
        frames_in_same_area = np.isin(event, frame_numbers_in_same_area)

        # Calculate the percentage of frames where the condition is met
        same_area_percentage = np.mean(frames_in_same_area)

        # Check if at least 60% of the frames in the event meet the condition
        if same_area_percentage >= 0.6:
            # Add the event to the filtered list
            filtered_chasing_events.append(event)


    min_centroid_speed = 25  # cm/s min speed for chasing
    min_both_centroid_speed = 35


    # Condition 4: speed is higher than threshold in both mice, and average speed is higher than threshold
    centroid_mouse0 = tracks[0, :, spine2_index, :]
    centroid_mouse1 = tracks[1, :, spine2_index, :]


    filtered_chases = []
    for sub_array in filtered_chasing_events:
        start = sub_array[0]-1
        end = sub_array[-1]
        # Clean up identity
        # Trim the centroid data to the frames we are currently considering
        centroid_mouse0_trimmed = centroid_mouse0[:, start:end]
        centroid_mouse1_trimmed = centroid_mouse1[:, start:end]
        # Initialize variables to hold the last known positions of each mouse (used to deal with NaN values in the tracking data)
        last_known_pos0 = centroid_mouse0_trimmed[:, 0]
        last_known_pos1 = centroid_mouse1_trimmed[:, 0]
        # Initialize arrays to hold the cleaned centroid data
        centroid_mouse0_cleaned = centroid_mouse0_trimmed.copy()
        centroid_mouse1_cleaned = centroid_mouse1_trimmed.copy()
        # Loop over the frames from the second frame to the last
        for i in range(1, end-start):
            if np.isnan(centroid_mouse0_trimmed[:, i]).any() and np.isnan(centroid_mouse1_trimmed[:, i]).any():
                continue
            # Calculate the Euclidean distance from each centroid in the current frame to each centroid in the previous frame
            dists = np.zeros((2, 2))
            dists[0, 0] = np.sqrt(np.sum((centroid_mouse0_trimmed[:, i] - last_known_pos0)**2))
            dists[0, 1] = np.sqrt(np.sum((centroid_mouse0_trimmed[:, i] - last_known_pos1)**2))
            dists[1, 0] = np.sqrt(np.sum((centroid_mouse1_trimmed[:, i] - last_known_pos0)**2))
            dists[1, 1] = np.sqrt(np.sum((centroid_mouse1_trimmed[:, i] - last_known_pos1)**2))
            if dists[0, 0] + dists[1, 1] <= dists[0, 1] + dists[1, 0]:
                last_known_pos0 = centroid_mouse0_trimmed[:, i]
                last_known_pos1 = centroid_mouse1_trimmed[:, i] 
            else:
                last_known_pos0 = centroid_mouse1_trimmed[:, i]
                last_known_pos1 = centroid_mouse0_trimmed[:, i]
                centroid_mouse0_cleaned[:, i], centroid_mouse1_cleaned[:, i] = centroid_mouse1_trimmed[:, i].copy(), centroid_mouse0_trimmed[:, i].copy()
        # Calculate centroid speed for each mouse
        mouse0_df = pd.DataFrame(centroid_mouse0_cleaned.T, columns=["x", "y"]).dropna()
        mouse1_df = pd.DataFrame(centroid_mouse1_cleaned.T, columns=["x", "y"]).dropna()
        dt_mouse0 = np.diff(mouse0_df.index.values*1000/FPS).astype(int) # ms
        dt_mouse1 = np.diff(mouse1_df.index.values*1000/FPS).astype(int) # ms
        dxy_mouse0 = mouse0_df[['x', 'y']].diff().values[1:]
        dxy_mouse1 = mouse1_df[['x', 'y']].diff().values[1:]
        mouse0_df = mouse0_df.iloc[1:]
        mouse1_df = mouse1_df.iloc[1:]
        mouse0_df["speed"] = np.linalg.norm(dxy_mouse0, axis=1) / dt_mouse0 / CM_TO_PX * 1000  # cm/s
        mouse1_df["speed"] = np.linalg.norm(dxy_mouse1, axis=1) / dt_mouse1 / CM_TO_PX * 1000  # cm/s
        mean_centroid0_speed = mouse0_df["speed"].mean()
        mean_centroid1_speed = mouse1_df["speed"].mean()
        mean_both_centroid_speed = np.mean([mean_centroid0_speed, mean_centroid1_speed])
        
        # Add to chasing list if both mice have a speed above the threshold
        if (mean_centroid0_speed > min_centroid_speed and mean_centroid1_speed > min_centroid_speed and mean_both_centroid_speed > min_both_centroid_speed):
            #print(mean_centroid0_speed, mean_centroid1_speed)
            filtered_chases.append(sub_array)

    return filtered_chases


def identify_chaser(filtered_chases, track_names, extremity_distances):
    # Chaser requirements
    # Chaser should be behind chased
    # So chaser is the mouse whose head is closer to te other mouse's tail
    #  So if extremidt_distance[3] < extremity_distance[2] then mouse 0 is chaser, else mouse 1 is chaser

    # get extremity idtances for the frames in chases
    chaser_ids = []
    chases = []
    
    for chase in filtered_chases:
        chase_extremity_distances = np.zeros((2, len(chase)))
        chase_extremity_distances[0] = extremity_distances[2, chase]
        chase_extremity_distances[1] = extremity_distances[3, chase]
        chaser = np.where(chase_extremity_distances[0] > chase_extremity_distances[1], 0, 1)
        # get which mouse was chaser for majority of indeces in case
        chaser_idx = np.bincount(chaser).argmax()
        chaser_id = track_names[chaser_idx]
        #store chaser ids in array
        chaser_ids.append(chaser_id)
        chases.append(chase)  
        
    return chases, chaser_ids

def export_results(chases, chaser_ids, video_path, export_directory):
    
    chase_data = {'start_frame' : [], 'end_frame' : [], 'start_timestamp' : [], 'end_timestamp' : [], 'duration_in_seconds' : [], 'chaser_id' : []}

    for subarray, chaser_id in zip(chases, chaser_ids):
        metadata_retrieval_matches = re.search(r'(.*?)(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}).*(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', video_path)
        arena_number_match = re.search(r'AEON(\d)', video_path)
        root = metadata_retrieval_matches.group(1)
        chunk_time = pd.to_datetime(metadata_retrieval_matches.group(3), format='%Y-%m-%dT%H-%M-%S')
        arena_number = arena_number_match.group(1)
        # Option to change the drive
        new_drive = "/ceph/aeon"
        root = re.sub(r'^.*?:', new_drive, root)

        start_frame = subarray[0]
        end_frame = subarray[-1]
        start_timestamp = chunk_time + pd.Timedelta(seconds=start_frame/FPS)
        end_timestamp = chunk_time + pd.Timedelta(seconds=end_frame/FPS)
        duration = (end_timestamp - start_timestamp).total_seconds()
        
        # Very short chases are likely to be false positives (errors in tracking)
        if duration > 3:
            chase_data['start_frame'].append(start_frame)
            chase_data['end_frame'].append(end_frame)
            chase_data['start_timestamp'].append(start_timestamp)
            chase_data['end_timestamp'].append(end_timestamp)
            chase_data['duration_in_seconds'].append(duration)
            
            # add chaser ids to chase_data
            chase_data['chaser_id'].append(chaser_id)

            # save a few sec before and after chase
            vid_export_dir = export_directory + '/all_chasing_videos/'
            if not os.path.exists(vid_export_dir):
                os.makedirs(vid_export_dir)
            
            extra_time = 1
            vid_start = start_timestamp - pd.Timedelta(seconds=extra_time)
            vid_end   = end_timestamp + pd.Timedelta(seconds=extra_time)
            frames_info = api.load(root, social02.CameraTop.Video, start=vid_start, end=vid_end)
            vid = video.frames(frames_info)
            save_path = vid_export_dir + "AEON" + arena_number + "_CameraTop_" + start_timestamp.strftime('%Y-%m-%dT%H-%M-%S') + "_" + end_timestamp.strftime('%Y-%m-%dT%H-%M-%S') + ".avi"
            video.export(vid, save_path, fps=FPS)
    chases_df = pd.DataFrame(chase_data)

    chases_df['start_timestamp'] = chases_df['start_timestamp'].apply(lambda x: x.strftime('%Y-%m-%dT%H-%M-%S'))
    chases_df['end_timestamp'] = chases_df['end_timestamp'].apply(lambda x: x.strftime('%Y-%m-%dT%H-%M-%S'))
    csv_path = vid_export_dir + "AEON" + arena_number + "_chases.csv"
    if not os.path.exists(csv_path):
        chases_df.to_csv(csv_path, index=False)
    else:
        existing_chases_df = pd.read_csv(csv_path)
        chases_df = pd.concat([existing_chases_df, chases_df]).drop_duplicates()
        chases_df.to_csv(csv_path, index=False)
        
    print(f"Videos and CSV exported to {vid_export_dir}")
    print(f"Found {chases_df.shape[0]} chase(s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 files to detect chases.")
    parser.add_argument('filepaths', nargs='+', help='List of file paths to process separated by spaces.')
    parser.add_argument("-e", "--export-directory", default="./exported_results", help="Directory to export results.")

    args = parser.parse_args()

    process_files(args.filepaths, args.export_directory) 
    
# run somethign like this: 
# python chasing_detection_all_files.py /ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraTop/predictions_social02/AEON3/analyses/CameraTop_2024-02-09T18-00-00_full_pose_id_all_frames.analysis.h5 /ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraTop/predictions_social02/AEON3/analyses/CameraTop_2024-02-13T16-00-00_full_pose_id_all_frames.analysis.h5 -e /ceph/aeon/aeon/code/scratchpad/Orsi