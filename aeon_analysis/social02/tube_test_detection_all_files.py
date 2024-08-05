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
        centroid_distances, extremity_distances, orientations, relative_distances, centroid_mouse0, centroid_mouse1 = compute_distances_and_orientations(tracks)
        
        # Extract possible tube test start frames
        possible_tube_test_starts = find_possible_tube_test_starts(orientations, centroid_distances, relative_distances, extremity_distances)
        
        # Filter start frames to only keep ones in corridor
        tube_test_starts, root, arena_number, chunk_time = filter_possible_tube_test_starts(possible_tube_test_starts, tracks, video_path)
        
        # Divide into consecutive tube test start events
        tube_test_start_arrays = get_start_arrays(tube_test_starts)
        
        # Get end frames, outcome, and export results
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)
        get_ends_and_export_results(tube_test_start_arrays, export_directory, orientations, centroid_distances, extremity_distances, tracks, track_names,root, chunk_time, arena_number, centroid_mouse0, centroid_mouse1)
        
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

    return centroid_distances, extremity_distances, orientations, relative_distances, centroid_mouse0, centroid_mouse1

def find_possible_tube_test_starts(orientations, centroid_distances, relative_distances, extremity_distances):
    angle_tolerance = 45
    max_distance = 50

    # Adjust the orientation of mouse 2
    adjusted_orientations = (orientations[1] + 180) % 360
    # this just did modulus operation on the orientation of the second mouse, so that it is in the same range as the first mouse. and turned around to make next line work

    # Condition 1: the mice have opposite orientations, within a certain tolerance
    orientation_condition = np.isclose(orientations[0], adjusted_orientations, atol=angle_tolerance)
    # Condition 2: the distance between the mice's centroids is less than a certain threshold, ensuring they are close to each other
    distance_condition = centroid_distances < max_distance
    # Condition 3: relative spine measure, removes cases where mice are side by side
    relative_distance_condition = relative_distances[1] > relative_distances[0]
    # Condition 4: the mice's tail-to-tail distance is greater than their nose-to-nose distance, removes cases where mice are back-to-back
    extremity_distance_condition = extremity_distances[1] > extremity_distances[0]
    # Find frames where all conditions are true
    tube_test_starts = np.where(np.logical_and.reduce([orientation_condition, distance_condition, relative_distance_condition, extremity_distance_condition]))
    
    return tube_test_starts

def filter_possible_tube_test_starts(possible_tube_test_starts, tracks, video_path):
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

    # Check if the distance is within the squared radii for both mice
    within_roi = (inner_radius**2 <= dist_squared) & (dist_squared <= outer_radius**2)

    # Check if the mice are in excluded regions
    in_excluded_region_nest = (spine2_x > center_x) & ((spine2_y >= nest_y1) & (spine2_y <= nest_y2))
    in_excluded_region_entrance = (spine2_x < center_x) & ((spine2_y >= entrance_y1) & (spine2_y <= entrance_y2))

    # Update the ROI condition to exclude the specified regions
    within_roi = within_roi & ~np.any(in_excluded_region_nest | in_excluded_region_entrance, axis=0)
    within_roi_both_mice = np.all(within_roi, axis=0)

    # Filter the frame numbers where both mice are within the ROI
    frame_numbers_in_roi = frame_numbers[within_roi_both_mice]

    # Filter possible tube test frames to only keep those where the mice are both within the corridor ROI
    tube_test_starts = np.intersect1d(possible_tube_test_starts, frame_numbers_in_roi)
    
    return tube_test_starts, root, arena_number, chunk_time

def get_start_arrays(tube_test_starts):
    max_frame_gap = 20
    min_tube_test_start_frames = 15

    # Divide possible_tube_test_starts into sub_arrays of consecutive frames (allowing for gaps up to a certain max)
    diffs = np.diff(tube_test_starts)
    indices = np.where(diffs > max_frame_gap)[0]
    indices += 1
    tube_test_starts = np.split(tube_test_starts, indices)

    # Filter sub_arrays to keep only those with more than a certain number of frames
    tube_test_start_arrays = [sub_array for sub_array in tube_test_starts if len(sub_array) > min_tube_test_start_frames]
    
    return tube_test_start_arrays

def get_ends_and_export_results(tube_test_start_arrays, export_directory, orientations, centroid_distances, extremity_distances, tracks, track_names,root, chunk_time, arena_number, centroid_mouse0, centroid_mouse1):
    spine2_index = 5
    angle_tolerance = 45    
    search_window_seconds = 1
    min_distance = 30
    max_distance = 60
    movement_threshold = 2
    vid_export_dir = export_directory + '/all_tube_test_videos/'
    tube_tests_data = {'start_frame': [], 'start_timestamp': [], 'end_frame': [], 'end_timestamp': [], 'duration_in_seconds': [], 'winner_id': []}

    for subarray in tube_test_start_arrays:
        # Check each possible_tube_test_starts frame interval for tracking errors
        # Skeleton flipping (i.e., tail end being mistaken for head) can lead to false tube test detections
        # Take all orientations in the interval, including frames that did not meet all the tube test start conditions
        all_start_orientations = orientations[:, subarray[0]:subarray[-1]+1]
        # Find how often the mice have the same orientation, within a certain tolerance
        orientation_condition = np.isclose(all_start_orientations[0], all_start_orientations[1], atol=angle_tolerance) 
        count = np.count_nonzero(orientation_condition)
        # Move to the next possible tube test start if skeleton flipping is detected
        if count > 1:
            continue

        # Find the possible tube test end frames
        first_possible_start_frame = subarray[0]
        last_possible_start_frame = subarray[-1]
        search_window = int(np.ceil(FPS*search_window_seconds))

        # Condition 1: the mice have the same orientations, within a certain tolerance
        orientation_condition = np.isclose(orientations[0, last_possible_start_frame:last_possible_start_frame + search_window], orientations[1, last_possible_start_frame:last_possible_start_frame + search_window], atol=angle_tolerance)
        # Condition 2: the distance between the mice's centroids is more than a certain threshold, removes cases where mice are fighting or side-by-side
        min_distance_condition = centroid_distances[last_possible_start_frame:last_possible_start_frame + search_window] > min_distance
        # Condition 3: the distance between the mice's centroids is less than a certain threshold, removes cases where mice "teleport" due to tracking errors
        max_distance_condition = centroid_distances[last_possible_start_frame:last_possible_start_frame + search_window] < max_distance 
        # Find frames where all conditions are true
        possible_tube_test_ends = last_possible_start_frame + np.where(np.logical_and.reduce([orientation_condition, min_distance_condition, max_distance_condition]))[0]
        # If there are frames where end conditions are met, clean identity tracking and check addtional conditions reliant on identity
        if len(possible_tube_test_ends) > 0:
            # Make list of frames where the identities are swapped
            # Trim the centroid data to the frames we are currently considering
            centroid_mouse0_trimmed = centroid_mouse0[:,first_possible_start_frame:last_possible_start_frame + search_window]
            centroid_mouse1_trimmed = centroid_mouse1[:,first_possible_start_frame:last_possible_start_frame + search_window]
            # Initialize variables to hold the last known positions of each mouse (used to deal with NaN values in the tracking data)
            last_known_pos0 = centroid_mouse0_trimmed[:, 0]
            last_known_pos1 = centroid_mouse1_trimmed[:, 0]
            # Initialize a list to hold the frames where the identities are swapped
            id_swaps = []
            # Loop over the frames from the second frame to the last
            for i in range(1, last_possible_start_frame + search_window - first_possible_start_frame):
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
                    id_swaps.append(i)
            # Possible quicker alternative:
            # centroid_mouse0_trimmed = centroid_mouse0[:, first_possible_start_frame:last_possible_start_frame + search_window - 1]  # all but the last frame
            # centroid_mouse1_trimmed_next_frame = centroid_mouse1[:, first_possible_start_frame + 1:last_possible_start_frame + search_window] # all but the first frame
            # shifted_centroid_dists = np.linalg.norm(centroid_mouse0_trimmed - centroid_mouse1_trimmed_next_frame, axis=0)
            # identity_swap = np.isclose(shifted_centroid_dists, 0, atol=10)
            # print(identity_swap)

            # Find which mouse turned around (loser)
            orientations_cleaned = orientations[:, first_possible_start_frame:last_possible_start_frame + search_window].copy()
            orientations_cleaned[:, id_swaps] = orientations_cleaned[::-1, id_swaps]
            start_orientations = orientations_cleaned[:, subarray-first_possible_start_frame]
            start_orientations = np.mean(start_orientations, axis=1)
            end_orientations = orientations_cleaned[:, possible_tube_test_ends-first_possible_start_frame]
            end_orientations = np.mean(end_orientations, axis=1)
            mouse_index = np.argmax(np.abs(start_orientations - end_orientations))
            winner_mouse_index = 1 - mouse_index
            winner_mouse_id = track_names[winner_mouse_index]
            # Condition 4: the loser is in front of the winner, removes cases where mouse A squeezes past mouse B, and mouse B turns around (false tube test detection)
            extremity_distances_cleaned = extremity_distances[:, first_possible_start_frame:last_possible_start_frame + search_window].copy()
            extremity_distances_cleaned[-2:, id_swaps] = extremity_distances_cleaned[-2:][::-1, id_swaps]
            mean_tail0_head1_distance = np.mean(extremity_distances_cleaned[2, possible_tube_test_ends-first_possible_start_frame])
            mean_tail1_head0_distance = np.mean(extremity_distances_cleaned[3, possible_tube_test_ends-first_possible_start_frame])
            front_mouse_condition = mean_tail0_head1_distance < mean_tail1_head0_distance if mouse_index == 0 else mean_tail1_head0_distance < mean_tail0_head1_distance
            # Condition 5: the loser's average movement between each frame is larger than a certain threshold, removes cases where the mice are stationary (e.g., grooming)
            tracks_cleaned = tracks[:, :, :, first_possible_start_frame:last_possible_start_frame + search_window].copy()
            tracks_cleaned[:, :, :, id_swaps] = tracks_cleaned[::-1, :, :, id_swaps]
            points_frame = tracks_cleaned[mouse_index, :, spine2_index, last_possible_start_frame-first_possible_start_frame:-1]  # all but the last frame
            points_next_frame = tracks_cleaned[mouse_index, :, spine2_index, last_possible_start_frame-first_possible_start_frame+1:]  # all but the first frame
            differences = points_next_frame - points_frame
            mean_movement = np.nanmean(np.linalg.norm(differences, axis=0))
            movement_condition = mean_movement > movement_threshold
            # Add tube test to final table if all end conditions are met
            if front_mouse_condition and movement_condition:
                start_timestamp = chunk_time + pd.Timedelta(seconds=first_possible_start_frame/FPS)
                tube_tests_data['start_frame'].append(first_possible_start_frame)
                tube_tests_data['start_timestamp'].append(start_timestamp)
                end_frame = possible_tube_test_ends[0]
                end_timestamp = chunk_time + pd.Timedelta(seconds=end_frame/FPS)
                tube_tests_data['end_frame'].append(end_frame)
                tube_tests_data['end_timestamp'].append(end_timestamp)
                tube_tests_data['duration_in_seconds'].append((end_timestamp - start_timestamp).total_seconds())
                tube_tests_data['winner_id'].append(winner_mouse_id)
                
                # Export video of the tube test for checking
                vid_start = start_timestamp - pd.Timedelta(seconds=5)
                vid_end   = end_timestamp + pd.Timedelta(seconds=5)
                frames_info = api.load(root, social02.CameraTop.Video, start=vid_start, end=vid_end)
                vid = video.frames(frames_info)
                save_path = vid_export_dir + "AEON" + arena_number + "_CameraTop_" + start_timestamp.strftime('%Y-%m-%dT%H-%M-%S') + ".avi"
                video.export(vid, save_path, fps=FPS)
                 
    # Save tube tests to csv
    tube_tests = pd.DataFrame(tube_tests_data)
    tube_tests['start_timestamp'] = tube_tests['start_timestamp'].apply(lambda x: x.strftime('%Y-%m-%dT%H-%M-%S'))
    tube_tests['end_timestamp'] = tube_tests['end_timestamp'].apply(lambda x: x.strftime('%Y-%m-%dT%H-%M-%S'))
    csv_path = vid_export_dir + "AEON" + arena_number + "_tube_tests.csv"
    if not os.path.exists(csv_path):
        tube_tests.to_csv(csv_path, index=False)
    else:
        existing_tube_tests = pd.read_csv(csv_path)
        tube_tests = pd.concat([existing_tube_tests, tube_tests]).drop_duplicates()
        tube_tests.to_csv(csv_path, index=False)
        
    print(f"Videos and CSV exported to {vid_export_dir}")
    print(f"Found {tube_tests.shape[0]} chase(s) in total")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 files to detect tube tests.")
    parser.add_argument('filepaths', nargs='+', help='List of file paths to process separated by spaces.')
    parser.add_argument("-e", "--export-directory", default="./exported_results", help="Directory to export results.")

    args = parser.parse_args()

    process_files(args.filepaths, args.export_directory) 
    
# run somethign like this: 
# python tube_test_detection_all_files.py /ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraTop/predictions_social02/AEON3/analyses/CameraTop_2024-02-09T18-00-00_full_pose_id_all_frames.analysis.h5 /ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraTop/predictions_social02/AEON3/analyses/CameraTop_2024-02-13T16-00-00_full_pose_id_all_frames.analysis.h5 -e /ceph/aeon/aeon/code/scratchpad/Orsi