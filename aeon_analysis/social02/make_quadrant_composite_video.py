import argparse
import numpy as np
import pandas as pd
import cv2
import random
from shapely.geometry import Point, Polygon
from pathlib import Path
from aeon.io import api, video
from aeon.schema.schemas import exp02, social02
import datajoint as dj
from aeon.dj_pipeline import streams
from aeon.dj_pipeline.analysis.block_analysis import *
from aeon.dj_pipeline import acquisition, streams


#global variables
CAMERA_A = 'CameraTop'
CAMERA_B_LIST = ['CameraSouth', 'CameraNorth', 'CameraEast', 'CameraWest']
CAMERA_DIMENSIONS = (1080, 1440)
TIMESTAMP_ERROR_TOLERANCE = pd.Timedelta(milliseconds=9) 
MAX_GAP_TO_FILL = pd.Timedelta(seconds=15)
NEGLIGIBLE_GAP = pd.Timedelta(milliseconds=100)
PART_RESTRICTION = {"part_name": "centroid"}
BASE_PATH = '/ceph/aeon/aeon/'
VIDEO_EXPORT_DIR = BASE_PATH + 'code/scratchpad/Orsi/pixel_mapping/composite_videos/'

def process_file(experiment, arena, dj_experiment_name, dj_chunk_start):

    KEY = {"experiment_name": dj_experiment_name}
    CHUNK_RESTRICTION = {"chunk_start": dj_chunk_start}
    ROOT = BASE_PATH + f'data/raw/{arena}/{experiment}/'

    # Load homographies
    homography_paths = [f'{BASE_PATH}code/scratchpad/Orsi/pixel_mapping/pixel_mapping_results/{experiment}/{arena}/H_{camera}.npy' for camera in CAMERA_B_LIST]
    homographies = [np.load(path) for path in homography_paths]

    # Fetch centroid data for Camera A
    start = pd.Timestamp(dj_chunk_start)
    end = start + pd.Timedelta(hours=1)
    # centroid_df = api.load(ROOT, exp02.CameraTop.Position, start, end)
    pose_query = (
        streams.SpinnakerVideoSource
        * tracking.SLEAPTracking.PoseIdentity.proj("identity_name", "identity_likelihood", anchor_part="part_name")
        * tracking.SLEAPTracking.Part
        & {"spinnaker_video_source_name": CAMERA_A}
        & KEY
        & CHUNK_RESTRICTION
        & PART_RESTRICTION
    )
    centroid_df = fetch_stream(pose_query)
    centroid_df.drop(columns=["spinnaker_video_source_name"], inplace=True)

    # Clean centroid data
    centroid_df = (
        centroid_df.groupby("identity_name", group_keys=False)
        .apply(lambda x: x.dropna().sort_index())
    )
    centroid_df = centroid_df.sort_index()
    centroid_df["x"], centroid_df["y"] = centroid_df["x"].astype(np.int32), centroid_df["y"].astype(np.int32)
    print(f"Centroid data loaded.")
    
    #Process centroid data to get quadrants
    transformed_corners_list = [get_transformed_corners(homography, CAMERA_DIMENSIONS) for homography in homographies]
    fov_centers = calculate_fov_centers(transformed_corners_list)
    centroid_df['possible_quadrants'] = find_quadrants_for_positions(centroid_df, transformed_corners_list)
    closest_quadrants = find_closest_quadrant(centroid_df, transformed_corners_list)
    centroid_df['possible_quadrants_extended'] = [
        row.possible_quadrants if row.possible_quadrants else closest_quadrants[idx]
        for idx, row in enumerate(centroid_df.itertuples())
    ]
    centroid_df['most_central_quadrant'] = find_most_central_quadrant(centroid_df, fov_centers)
    centroid_df['both_mice_quadrants'] = find_common_quadrants_for_both_mice(centroid_df)
    
    # Finalize and clean up DataFrame
    selected_quadrants_df = determine_quadrant_camera(centroid_df)
    centroid_df = centroid_df.merge(selected_quadrants_df, left_index=True, right_index=True, how='left')
    centroid_df['selected_quadrant'] = centroid_df['selected_quadrant'].apply(lambda x: sorted(set(x)))

    # Populate DataFrame with timestamps and selected quadrants
    quadrant_timestamp_df = centroid_df.groupby(centroid_df.index).first().reset_index()[['time']]
    quadrant_timestamp_df.set_index('time', inplace=True)
    # Add selected quadrant to the DataFrame
    quadrant_timestamp_df['selected_quadrant'] = centroid_df.groupby(centroid_df.index)['selected_quadrant'].first()
    # Check for NaN values and print debug information
    nan_indices = quadrant_timestamp_df['selected_quadrant'].isna()
    if nan_indices.any():
        print("NaN values found in selected_quadrant at indices:", quadrant_timestamp_df[nan_indices].index.tolist())
    # Ensure all selected_quadrant values are lists of integers
    quadrant_timestamp_df['selected_quadrant'] = quadrant_timestamp_df['selected_quadrant'].apply(
        lambda x: [int(q) for q in x] if isinstance(x, list) else ([int(x)] if pd.notna(x) else x)
    )
    # Map the quadrant index to the camera name based on the order in CAMERA_B_LIST
    quadrant_to_camera_name = {i: camera for i, camera in enumerate(CAMERA_B_LIST)}
    # Add the selected camera name to the DataFrame, handling NaN values
    quadrant_timestamp_df['selected_camera_name'] = quadrant_timestamp_df['selected_quadrant'].apply(
        lambda x: [quadrant_to_camera_name[q] for q in x] if isinstance(x, list) and pd.notna(x).all() else []
    )
    print("Quadrant camera selection completed.")

    # Fill in missing timestamps in data if they are shorter than the threshold and flanked by the same quadrant
    # Generate a complete range of timestamps for the expected frame rate (50 fps)
    start_time = pd.Timestamp(dj_chunk_start)
    end_time = start_time + pd.Timedelta(hours=1)
    expected_timestamps = pd.date_range(start=start_time, end=end_time, freq='19.999981ms')

    # Create a DataFrame for the expected timestamps with NaN values
    expected_df = pd.DataFrame(index=expected_timestamps)

    # Merge the existing data with the expected timestamps
    quadrant_timestamp_df = pd.merge_asof(
        expected_df.reset_index().rename(columns={'index': 'time'}),
        quadrant_timestamp_df.reset_index().rename(columns={'index': 'time'}),
        on='time',
        direction='nearest',
        tolerance=TIMESTAMP_ERROR_TOLERANCE
    ).set_index('time')

    # Identify runs of NaNs that are max gap to fill seconds or less
    is_nan = quadrant_timestamp_df['selected_camera_name'].isna()
    nan_runs = is_nan.astype(int).groupby((~is_nan).cumsum()).cumsum()

    # Find the start and end of each NaN run
    nan_run_starts = nan_runs[is_nan & (nan_runs == 1)].index
    nan_run_ends = nan_runs[is_nan & (nan_runs == nan_runs.groupby((~is_nan).cumsum()).transform('max'))].index

    # Process NaN runs for filling
    for start, end in zip(nan_run_starts, nan_run_ends):
        prev_timestamp = start - pd.Timedelta(milliseconds=1)
        next_timestamp = end + pd.Timedelta(milliseconds=1)
        
        # Find the closest previous and next timestamps in the DataFrame
        closest_prev_timestamp = quadrant_timestamp_df.index.asof(prev_timestamp)
        next_index = quadrant_timestamp_df.index.searchsorted(next_timestamp)
        closest_next_timestamp = quadrant_timestamp_df.index[next_index] if next_index < len(quadrant_timestamp_df.index) else None
        
        prev_camera = quadrant_timestamp_df.loc[closest_prev_timestamp, 'selected_camera_name'] if closest_prev_timestamp else 'Unknown'
        next_camera = quadrant_timestamp_df.loc[closest_next_timestamp, 'selected_camera_name'] if closest_next_timestamp else 'Unknown'
        
        prev_quadrant = quadrant_timestamp_df.loc[closest_prev_timestamp, 'selected_quadrant'] if closest_prev_timestamp else 'Unknown'

        # Print debug information for each gap
        gap_length = is_nan[start:end].sum()
        #print(f"Gap from {start} to {end} with previous camera {prev_camera} and next camera {next_camera}")
        print(f"Length of gap: {gap_length} frames")
        
        # Fill NaNs if previous and next cameras are the same and the gap is below the threshold
        if ((end - start <= MAX_GAP_TO_FILL) and (prev_camera == next_camera) and (prev_camera != 'Unknown')) or (end - start <= NEGLIGIBLE_GAP):
            fill_camera = prev_camera if isinstance(prev_camera, list) else [prev_camera]
            fill_quadrant = prev_quadrant if isinstance(prev_quadrant, list) else [prev_quadrant]
            
            # Fill the gap in the DataFrame
            quadrant_timestamp_df.loc[start:end, 'selected_camera_name'] = [fill_camera] * len(quadrant_timestamp_df.loc[start:end])
            quadrant_timestamp_df.loc[start:end, 'selected_quadrant'] = [fill_quadrant] * len(quadrant_timestamp_df.loc[start:end])
        else:
            raise ValueError("Some frames were not assigned quadrant cameras and the gap could not be filled.")

            
            
    print("Timestamp gaps filled in.")
    
    # Stich together the video
    # Step 1: make a frames_info for each quadrant camera as dict

    # Initialize a dictionary to hold frames information for each camera
    frames_info_dict = {}

    # Loop through each camera name and load the data
    for camera in CAMERA_B_LIST:
        # NOTE: this doesnt seem to work if I don't define these frame infos manually like this 
        if camera == 'CameraNorth':
            frames_info = api.load(ROOT, social02.CameraNorth.Video, start=start_time, end=end_time)
        elif camera == 'CameraSouth':
            frames_info = api.load(ROOT, social02.CameraSouth.Video, start=start_time, end=end_time)
        elif camera == 'CameraEast':
            frames_info = api.load(ROOT, social02.CameraEast.Video, start=start_time, end=end_time)
        elif camera == 'CameraWest':
            frames_info = api.load(ROOT, social02.CameraWest.Video, start=start_time, end=end_time)

        # Store the loaded frames info in the dictionary
        frames_info_dict[camera] = frames_info
        
        
    # Step 2: to make frames_info_final, loop over quadrant_timestamp_df tiemstamps and get the corresponding frames_info for the selected camera
    # Initialize an empty DataFrame to store all the frames
    frames_info_final = pd.DataFrame()

    # Loop through each timestamp in quadrant_timestamp_df
    for index, row in quadrant_timestamp_df.iterrows():
        timestamp = index  # Current timestamp from the DataFrame
        selected_cameras = row['selected_camera_name']  # Cameras to get frames from for this timestamp
        
        # For each selected camera, get the frames closest to the current timestamp
        for camera_name in selected_cameras:
            frames_info = frames_info_dict[camera_name]  # Access preloaded frame info for the camera
            
            # Find the closest frame to the current timestamp within the time tolerance
            closest_frame_info = frames_info.loc[
                (frames_info.index >= (timestamp - TIMESTAMP_ERROR_TOLERANCE)) & 
                (frames_info.index <= (timestamp + TIMESTAMP_ERROR_TOLERANCE))
            ]
            
            # Append these frames to the final DataFrame
            frames_info_final = pd.concat([frames_info_final, closest_frame_info], ignore_index=True)

    # Sort the final DataFrame by time if necessary
    frames_info_final.sort_index(inplace=True)

    # Save the frames info to csv
    experiment = experiment.replace('.', '')
    start_time_str = start_time.strftime("%Y-%m-%dT%H-%M-%S")
    frames_info_final.to_csv(f'{VIDEO_EXPORT_DIR}/{arena}_{experiment}_{start_time_str}_composite_vid_frames_info.csv')

    # Step 3: call video.frames on this to compile video and save it
    vid = video.frames(frames_info_final)
    # save the video
    save_path = VIDEO_EXPORT_DIR + f"{arena}_{experiment}_{start_time_str}_composite_video.avi"
    video.export(vid, save_path, fps=50)
    print(f"Composite video saved to: {save_path}")
    
    

def get_transformed_corners(homography_matrix, img_shape):
    h, w = img_shape
    corners = np.array([[0, 0, 1], [0, h - 1, 1], [w - 1, 0, 1], [w - 1, h - 1, 1]])
    transformed_corners = (homography_matrix @ corners.T).T
    transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2][:, np.newaxis]

    if len(transformed_corners) == 4:
        sorted_corners = sorted(transformed_corners, key=lambda point: (point[0], point[1]))
        top_left, bottom_left = sorted(sorted_corners[:2], key=lambda point: point[1])
        top_right, bottom_right = sorted(sorted_corners[2:], key=lambda point: point[1])
        return np.array([top_left, top_right, bottom_right, bottom_left])
    else:
        print(f"Only {len(transformed_corners)} valid corners found.")
        return transformed_corners

def calculate_fov_centers(transformed_corners_list):
    centers = [
        (np.mean(corners[:, 0]), np.mean(corners[:, 1])) if len(corners) == 4 else None
        for corners in transformed_corners_list
    ]
    return centers

def find_quadrants_for_positions(positions_df, transformed_corners_list):
    possible_quadrants = []
    for _, row in positions_df.iterrows():
        point = Point(row['x'], row['y'])
        in_fovs = [
            i for i, corners in enumerate(transformed_corners_list)
            if len(corners) == 4 and Polygon(corners).contains(point)
        ]
        possible_quadrants.append(in_fovs)
    return possible_quadrants

def find_closest_quadrant(positions_df, transformed_corners_list):
    closest_quadrants = []
    for _, row in positions_df.iterrows():
        point = Point(row['x'], row['y'])
        distances = [
            (np.sqrt((row['x'] - np.mean(corners[:, 0])) ** 2 + (row['y'] - np.mean(corners[:, 1])) ** 2), i)
            for i, corners in enumerate(transformed_corners_list) if len(corners) == 4
        ]
        distances.sort()
        closest_quadrants.append([distances[0][1]] if distances else None)
    return closest_quadrants

def find_most_central_quadrant(positions_df, fov_centers):
    most_central_quadrant = []
    for _, row in positions_df.iterrows():
        quadrants = row['possible_quadrants_extended']
        if not quadrants:
            most_central_quadrant.append(None)
            continue
        distances = [
            (np.sqrt((row['x'] - fov_centers[q][0]) ** 2 + (row['y'] - fov_centers[q][1]) ** 2), q)
            for q in quadrants
        ]
        distances.sort()
        most_central_quadrant.append(distances[0][1])
    return most_central_quadrant

def find_common_quadrants_for_both_mice(positions_df):
    both_mice_quadrants = []
    grouped = positions_df.groupby(positions_df.index)

    for timestamp, group in grouped:
        if len(group) < 2:
            both_mice_quadrants.append((timestamp, None))
            continue

        quadrants_1 = set(group.iloc[0]['possible_quadrants'])
        quadrants_2 = set(group.iloc[1]['possible_quadrants'])
        common_quadrants = list(quadrants_1.intersection(quadrants_2))

        both_mice_quadrants.append((timestamp, common_quadrants))

    common_quadrants_df = pd.DataFrame(both_mice_quadrants, columns=['time', 'both_mice_quadrants'])
    common_quadrants_df.set_index('time', inplace=True)
    return common_quadrants_df



def determine_quadrant_camera(positions_df):
    selected_quadrants = []
    for timestamp, group in positions_df.groupby(positions_df.index):
        common_quadrants = group['both_mice_quadrants'].iloc[0]
        if not common_quadrants:
            selected_quadrant = group['most_central_quadrant'].tolist()
        elif len(common_quadrants) == 1:
            selected_quadrant = [int(common_quadrants[0])]
        else:
            central_quadrants = group['most_central_quadrant'].unique()
            intersection = [int(q) for q in central_quadrants if q in common_quadrants]
            if len(intersection) == 1:
                selected_quadrant = intersection
            elif len(intersection) == 2:
                most_central_counts = group['most_central_quadrant'].value_counts()
                if most_central_counts[intersection[0]] > 0 and most_central_counts[intersection[1]] > 0:
                    selected_quadrant = [random.choice(intersection)]
                else:
                    selected_quadrant = most_central_counts.idxmax()
            else:
                selected_quadrant = list(group['most_central_quadrant'].mode().values)
        selected_quadrants.append((timestamp, selected_quadrant))
    return pd.DataFrame(selected_quadrants, columns=['time', 'selected_quadrant']).set_index('time')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make composite videos of quadrant cameras from a 1h chunk(s).")
    parser.add_argument('--experiments', type=str, nargs='+', help='Names of the experiments, separated by spaces.')
    parser.add_argument('--arenas', type=str, nargs='+', help='Arena names, separated by spaces.')
    parser.add_argument('--dj_experiment_names', type=str, nargs='+', help='Experiment-arena names needed for DJ query, separated by spaces.')
    parser.add_argument('--dj_chunk_starts', type=str, nargs='+', help='Chunk start times in the format "YYYY-MM-DD HH:MM:SS", separated by spaces.')

    args = parser.parse_args()

    # Check if the number of provided arguments is consistent
    if not (len(args.experiments) == len(args.arenas) == len(args.dj_experiment_names) == len(args.dj_chunk_starts)):
        print("Error: The number of experiments, arenas, dj_experiment_names, and dj_chunk_starts must be equal.")
        exit(1)

    # Process each file
    for experiment, arena, dj_experiment_name, dj_chunk_start in zip(args.experiments, args.arenas, args.dj_experiment_names, args.dj_chunk_starts):
        process_file(experiment, arena, dj_experiment_name, dj_chunk_start)


