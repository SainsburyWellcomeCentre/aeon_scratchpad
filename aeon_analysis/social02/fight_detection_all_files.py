import argparse
import os
import re

import h5py
import pandas as pd
import numpy as np
from pathlib import Path

from aeon.io import video
from aeon.io import api
from aeon.schema.schemas import exp02, social02

# Globals
FPS = 50
CM_TO_PX = 5.4  # 1 cm = 5.4 px

def process_files(filepaths, centroid_data_path, export_directory):
    """
    Process each HDF5 file to extract tracks, compute distances, identify fights, and export results.

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
        
        try:
            ## Funtions that do the processing:
            # Extract tracks from h5
            tracks, track_occupancy, node_names, edge_inds, video_path = extract_tracks(filepath)
            
            # Extract centroid data using api
            centroid_blob_data = load_centroid_data(centroid_data_path, filepath)
            
            # Comopute distances,
            results = compute_distances(tracks, node_names, edge_inds)
            centroid_distances_ffill, centroid_mouse0, centroid_mouse1, nose_head_distances_mouse0, nose_head_distances_mouse1, mean_interspinal_distances_mouse0, mean_interspinal_distances_mouse1 = results
            
            # Compute blob speed
            centroid_blob_data = calculate_blob_speed(centroid_blob_data)
            
            # Extract possible fighting frames
            possible_fight_frames, cond1_frames = find_possible_fight_frames(centroid_blob_data, centroid_distances_ffill, 
                                                                            nose_head_distances_mouse0, nose_head_distances_mouse1, 
                                                                            mean_interspinal_distances_mouse0, mean_interspinal_distances_mouse1)
        
            # Extract possible fighting events
            possible_fight_events = get_possible_fight_events(possible_fight_frames, track_occupancy, cond1_frames)
            
            # Filte events based on individual speed
            fight_events = filter_events(possible_fight_events, centroid_mouse0, centroid_mouse1)
            
            # Export results and further restrict length of events
            if not os.path.exists(export_directory):
                os.makedirs(export_directory)
            export_results(fight_events, video_path, export_directory)
            
            print(f"Completed processing for file: {filepath}\n")
            
        except Exception as e:
            print(f"An error occurred while processing file {filepath}: {e}")
        

def extract_tracks(filepath):
    # Open the HDF5 file in read mode
    with h5py.File(filepath, 'r') as f:
        # Extract the tracks
        tracks = f['tracks'][:]
        track_occupancy = f['track_occupancy'][:]
        node_names = f['node_names'][:].astype(str)
        edge_inds = f['edge_inds'][:]
        video_path = f['video_path'][()].decode('utf-8')
        print(video_path)

    return tracks, track_occupancy,  node_names, edge_inds, video_path

def load_centroid_data(centroid_data_path, filepath):
    """Load centroid blob data."""
    # Extract the start time from the filename
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', filepath)
    if match:
        start_str = match.group(1).replace('T', ' ')
        start = pd.Timestamp(start_str).tz_localize(None)
        end = start + pd.Timedelta(hours=1)
    else:
        raise ValueError("Filename does not contain a valid datetime format")
    centroid_blob_data = api.load(Path(centroid_data_path), exp02.CameraTop.Position, start, end)
    centroid_blob_data.reset_index(inplace=True)
    centroid_blob_data.dropna(inplace=True)
    
    return centroid_blob_data

def compute_distances(tracks, node_names, edge_inds):
    # Constants
    spine2_index = np.where(node_names == 'spine2')[0][0]

    # Centroid distances 
    centroid_mouse0 = tracks[0, :, spine2_index, :]
    centroid_mouse1 = tracks[1, :, spine2_index, :]
    centroid_distances = np.linalg.norm(centroid_mouse0 - centroid_mouse1, axis=0)
    centroid_distances_ffill = pd.Series(centroid_distances).ffill().to_numpy()

    # Internode distances
    internode_distances_mouse0 = np.zeros((len(edge_inds), tracks.shape[3]))
    internode_distances_mouse1 = np.zeros((len(edge_inds), tracks.shape[3]))
    for i, node_pair in enumerate(edge_inds):
        internode_distances_mouse0[i] = np.linalg.norm(tracks[0, :, node_pair[0], :] - tracks[0, :, node_pair[1], :], axis=0)
        internode_distances_mouse1[i] = np.linalg.norm(tracks[1, :, node_pair[0], :] - tracks[1, :, node_pair[1], :], axis=0)
    nose_head_distances_mouse0 = internode_distances_mouse0[0,:]
    nose_head_distances_mouse1 = internode_distances_mouse1[0,:]
    mean_interspinal_distances_mouse0 = np.mean(internode_distances_mouse0[3:,:], axis=0)
    mean_interspinal_distances_mouse1 = np.mean(internode_distances_mouse1[3:,:], axis=0)

    return centroid_distances_ffill,centroid_mouse0, centroid_mouse1, nose_head_distances_mouse0, nose_head_distances_mouse1, mean_interspinal_distances_mouse0, mean_interspinal_distances_mouse1

def calculate_blob_speed(centroid_blob_data):
    # Blob speed
    dxy = centroid_blob_data[["x", "y"]].diff().values[1:]
    dt = (np.diff(centroid_blob_data["time"]) / 1e6).astype(int)  # ms
    centroid_blob_data["speed"] = np.concatenate(([0], np.linalg.norm(dxy, axis=1) / dt / CM_TO_PX * 1000))  # cm/s
    k = np.ones(10) / 10  # running avg filter kernel (10 frames)
    centroid_blob_data["speed"] = np.convolve(centroid_blob_data["speed"], k, mode="same")
    
    return centroid_blob_data

def find_possible_fight_frames(centroid_blob_data, centroid_distances_ffill, nose_head_distances_mouse0, nose_head_distances_mouse1, mean_interspinal_distances_mouse0, mean_interspinal_distances_mouse1):
    max_distance = 20 # px
    max_nose_head_distance = 7 # px
    max_interspinal_distance = 10 # px
    min_blob_speed = 3  # cm/s 

    # Condition 1: the mice are close to each other
    cond1_frames = np.where(centroid_distances_ffill < max_distance)[0]

    # Condition 2: the mean internode distances are within a certain range
    # Condition 2a: the distance between the mice's noses and heads is within a certain range
    cond2a = np.logical_or(nose_head_distances_mouse0 > max_nose_head_distance, nose_head_distances_mouse1 > max_nose_head_distance)
    # Condition 2b: the mean distance between the mice's own spine nodes is within a certain range
    cond2b = np.logical_or(mean_interspinal_distances_mouse0 > max_interspinal_distance, mean_interspinal_distances_mouse1 > max_interspinal_distance)
    # Find frames where conditions 2a or 2b are true
    cond2 = np.logical_or(cond2a, cond2b)
    cond2_frames = np.where(cond2)[0]

    # Condition 3: the speed of the blob is above a certain threshold
    cond3_frames = centroid_blob_data[(centroid_blob_data["speed"] > min_blob_speed)].index.values

    possible_fight_frames = np.intersect1d(np.intersect1d(cond1_frames, cond2_frames), cond3_frames)
    
    return possible_fight_frames, cond1_frames

def get_possible_fight_events(possible_fight_frames, track_occupancy, cond1_frames):
    max_frame_gap = FPS*4
    min_num_frames = int(FPS*0.1)

    # Divide possible_tube_test_starts into sub_arrays of consecutive frames (allowing for gaps up to a certain max)
    diffs = np.diff(possible_fight_frames)
    indices = np.where(diffs > max_frame_gap)[0]
    indices += 1
    possible_fight_frames = np.split(possible_fight_frames, indices)
    # Filter sub_arrays to keep only those with more than a certain number of frames
    possible_fight_frames = [sub_array for sub_array in possible_fight_frames if len(sub_array) > min_num_frames]

    max_frame_gap = FPS*2
    # Include empty frames where the mice were close to each other in the previous frame they were detected
    # If these occur close to or during the time of a possible fight, it's likely the mice are fighting and not detected due to weird poses
    # These frames will have been dropped by condition 2 but can help connect/extend the possible fights detected above
    empty_frames = np.where(np.where((track_occupancy[:, 0] == 0) & (track_occupancy[:, 1] == 0), 1, 0))[0]
    empty_frames = np.intersect1d(cond1_frames, empty_frames) # Only select empty frames where the mice were previously close to each other
    possible_fight_frames = np.concatenate(possible_fight_frames)
    possible_fights_w_empty_frames = np.union1d(possible_fight_frames, empty_frames)
    diffs = np.diff(possible_fights_w_empty_frames)
    indices = np.where(diffs > max_frame_gap)[0]
    indices += 1
    possible_fights_w_empty_frames = np.split(possible_fights_w_empty_frames, indices)
    # Only keep the subarrays that contain at least one frame from the original possible_fights array
    # i.e., don't include subarrays entirely composed of empty frames
    check = [any(frame in possible_fight_frames for frame in sub_array) for sub_array in possible_fights_w_empty_frames]
    possible_fight_events = [possible_fights_w_empty_frames[i] for i, val in enumerate(check) if val]
    
    return possible_fight_events

def filter_events(possible_fight_events, centroid_mouse0, centroid_mouse1):
    min_centroid_speed = 20  # cm/s min speed for fighting
    min_both_centroid_speed = 15

    fight_events = []
    for sub_array in possible_fight_events:
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
        # Add to fights list if either of the mice have a speed above the threshold
        if (mean_centroid0_speed > min_centroid_speed or mean_centroid1_speed > min_centroid_speed or mean_both_centroid_speed > min_both_centroid_speed):
            # print(mouse1_df)
            fight_events.append(sub_array)
            
    return fight_events

def export_results(fight_events, video_path, export_directory):
    vid_export_dir = export_directory + '/all_fighting_videos/'    
    fight_data = {'start_frame' : [], 'end_frame' : [], 'start_timestamp' : [], 'end_timestamp' : [], 'duration_in_seconds' : []}

    for subarray in fight_events:
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
        # Very short fights are likely to be false positives (errors in tracking)
        if duration > 1:
            fight_data['start_frame'].append(start_frame)
            fight_data['end_frame'].append(end_frame)
            fight_data['start_timestamp'].append(start_timestamp)
            fight_data['end_timestamp'].append(end_timestamp)
            fight_data['duration_in_seconds'].append(duration)
            
            vid_start = start_timestamp - pd.Timedelta(seconds=1)
            vid_end   = end_timestamp + pd.Timedelta(seconds=1)
            frames_info = api.load(root, social02.CameraTop.Video, start=vid_start, end=vid_end)
            vid = video.frames(frames_info)
            save_path = vid_export_dir + "AEON" + arena_number + "_CameraTop_" + start_timestamp.strftime('%Y-%m-%dT%H-%M-%S') + "_" + end_timestamp.strftime('%Y-%m-%dT%H-%M-%S') + ".avi"
            video.export(vid, save_path, fps=FPS)
    fights_df = pd.DataFrame(fight_data)
    fights_df['start_timestamp'] = fights_df['start_timestamp'].apply(lambda x: x.strftime('%Y-%m-%dT%H-%M-%S'))
    fights_df['end_timestamp'] = fights_df['end_timestamp'].apply(lambda x: x.strftime('%Y-%m-%dT%H-%M-%S'))
    csv_path = vid_export_dir + "AEON" + arena_number + "_fights.csv"
    if not os.path.exists(csv_path):
        fights_df.to_csv(csv_path, index=False)
    else:
        existing_fights_df = pd.read_csv(csv_path)
        fights_df = pd.concat([existing_fights_df, fights_df]).drop_duplicates()
        fights_df.to_csv(csv_path, index=False)
        
    print(f"Videos and CSV exported to {vid_export_dir}")
    print(f"Found {fights_df.shape[0]} chase(s) in total")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 files to detect fights.")
    parser.add_argument('filepaths', nargs='+', help='List of file paths to process separated by spaces.')
    parser.add_argument("-c", "--centroid-data-path", default="/ceph/aeon/aeon/data/raw/AEON4/social0.2", help="Path to the raw data (centroid).")
    parser.add_argument("-e", "--export-directory", default="./exported_results", help="Directory to export results.")

    args = parser.parse_args()

    process_files(args.filepaths, args.centroid_data_path , args.export_directory) 
    
# Example usage:
#python fight_detection_all_files.py /ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraTop/predictions_social02/AEON4/analyses/CameraTop_2024-02-17T18-00-00_full_pose_id_all_frames.analysis.h5 -c /ceph/aeon/aeon/data/raw/AEON4/social0.2 -e /ceph/aeon/aeon/code/scratchpad/Orsi