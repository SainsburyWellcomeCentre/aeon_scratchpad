import os
import pandas as pd
from pathlib import Path


def save_data_to_parquet(
    df: pd.DataFrame,
    experiment_name: str,
    period_name: str,
    data_type: str,
    data_dir: Path
) -> Path:
    """Saves any DataFrame to a parquet file with consistent naming and metadata.

    Args:
        df (pd.DataFrame): Data to save
        experiment_name (str): Name of the experiment (use 'combined' for multi-experiment data)
        period_name (str): Period name (presocial, social, postsocial)
        data_type (str): Type of data (position, patch, foraging, rfid, sleep, explore)
        data_dir (Path): Directory to save the file

    Returns:
        Path: Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Add period column for reference if not already present
    df = df.copy()
    if 'period' not in df.columns:
        df['period'] = period_name
    
    # Handle index properly for consistent loading
    if df.index.name and df.index.name != 'time':
        df = df.reset_index()

    # Create filename
    filename = f"{experiment_name}_{period_name}_{data_type}.parquet"
    file_path = data_dir / filename

    print(f"  Saving to {file_path}...")
    # Save to parquet with compression
    df.to_parquet(file_path, compression="snappy")

    # Report file stats
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"  Saved successfully: {len(df)} rows, {file_size_mb:.2f} MB on disk")

    return file_path


def load_data_from_parquet(
    experiment_name: str | None,
    period: str | None,
    data_type: str,
    data_dir: Path,
    set_time_index: bool = False
) -> pd.DataFrame:
    """Loads saved data from parquet files.

    Args:
        experiment_name (str, optional): Filter by experiment name. If None, load all experiments.
        period (str, optional): Filter by period (presocial, social, postsocial). If None, load all periods.
        data_type (str): Type of data to load (position, patch, foraging, rfid, sleep, explore)
        data_dir (Path): Directory containing parquet files.
        set_time_index (bool, optional): If True, set 'time' column as DataFrame index.

    Returns:
        pd.DataFrame: Combined DataFrame of all matching parquet files.
    """
    if not data_dir.exists():
        print(f"Directory {data_dir} does not exist. No data files found.")
        return pd.DataFrame()

    # Create pattern based on filters
    pattern = ""
    if experiment_name:
        pattern += f"{experiment_name}_"
    else:
        pattern += "*_"

    if period:
        pattern += f"{period}_"
    else:
        pattern += "*_"

    pattern += f"{data_type}.parquet"

    # Find matching files
    matching_files = list(data_dir.glob(pattern))

    if not matching_files:
        print(f"No matching data files found with pattern: {pattern}")
        return pd.DataFrame()

    print(f"Found {len(matching_files)} matching files")

    # Load and concatenate matching files
    dfs = []
    total_rows = 0
    for file in matching_files:
        print(f"Loading {file}...")
        df = pd.read_parquet(file)
        total_rows += len(df)
        dfs.append(df)
        print(f"  Loaded {len(df)} rows")

    # Combine data
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        if set_time_index and 'time' in combined_df.columns:
            combined_df = combined_df.set_index('time')
        print(f"Combined data: {len(combined_df)} rows")
        return combined_df
    else:
        return pd.DataFrame()


def save_all_experiment_data(
    experiments: list,
    data_dict: dict,
    data_type: str,
    data_dir: Path
) -> None:
    """Save data for all experiments and periods in a standardized way.
    
    Args:
        experiments (list): List of experiment dictionaries with 'name' field
        data_dict (dict): Nested dictionary with structure {exp_name: {period_name: dataframe}}
        data_type (str): Type of data (position, patch, foraging, rfid, sleep, explore)
        data_dir (Path): Directory to save files
    """
    # Save individual experiment data
    for exp in experiments:
        for period in ['presocial', 'social', 'postsocial']:
            df = data_dict[exp['name']][period]
            if not df.empty:
                save_data_to_parquet(
                    df,
                    exp['name'],
                    period,
                    data_type,
                    data_dir
                )
    
    # Save combined data for each period
    for period in ['presocial', 'social', 'postsocial']:
        period_dfs = []
        for exp in experiments:
            df = data_dict[exp['name']][period]
            if not df.empty:
                period_dfs.append(df)
        
        if period_dfs:
            combined_df = pd.concat(period_dfs, ignore_index=True)
            save_data_to_parquet(
                combined_df,
                'combined',
                period,
                data_type,
                data_dir
            )