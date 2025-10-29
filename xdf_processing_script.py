#!/usr/bin/env python3
"""
XDF Stream Processing Utilities
================================

This module provides utilities for extracting, filtering, synchronizing, and
resampling multi-modal data streams from XDF (Extensible Data Format) files.

Key Features:
- Robust channel and stream name matching (whitespace-tolerant)
- Multi-stream synchronization and resampling
- Flexible data truncation and interpolation
- Diagnostic tools for debugging stream selection
- Complete all-in-one pipeline function

Typical Workflow:
1. Load streams from XDF file using pyxdf
2. Inspect available streams with print_stream_info()
3. Define desired streams/channels in a selection dictionary
4. Use process_xdf_streams() for complete pipeline, OR
5. Use individual functions for custom processing

Author: [Your Name]
Date: 2025-10-10
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


# ===========================================================
# Channel Label Extraction
# ===========================================================

def extract_channel_labels(stream):
    """
    Extract channel labels, signal types, and units from a stream's metadata.

    This function navigates the nested XDF metadata structure to retrieve
    channel information. It handles various metadata formats and provides
    fallback behavior when metadata is incomplete or malformed.

    Parameters
    ----------
    stream : dict
        A stream dictionary from an XDF file, containing at minimum:
        - stream['info']: Metadata dictionary
        - stream['info']['desc']: Description containing channel information
        - stream['time_series']: The actual data array

    Returns
    -------
    labels : list of str
        Channel labels/names
    types : list of str
        Signal types for each channel (empty strings if unavailable)
    units : list of str
        Measurement units for each channel (empty strings if unavailable)

    Notes
    -----
    - If metadata extraction fails, generates generic labels: "Channel 0", "Channel 1", etc.
    - Assumes channel information is stored in:
      stream['info']['desc'][0]['channels'][0]['channel']
    - Each channel entry should be a dict with 'label', 'type', and 'unit' keys
    """
    try:
        # Navigate the nested metadata structure to find channel entries
        channel_entries = stream['info']['desc'][0]['channels'][0]['channel']

        # Handle both single channel (dict) and multiple channels (list of dicts)
        if not isinstance(channel_entries, list):
            channel_entries = [channel_entries]

        labels, types, units = [], [], []

        # Extract information from each channel entry
        for i, ch in enumerate(channel_entries):
            # Get label (try 'label' first, fallback to 'name', finally generic)
            label = ch.get('label', ch.get('name', [f"Channel {i}"]))[0]
            signal_type = ch.get('type', [''])[0]
            unit = ch.get('unit', [''])[0]

            labels.append(label)
            types.append(signal_type)
            units.append(unit)

        return labels, types, units

    except (KeyError, IndexError, TypeError):
        # Fallback when metadata is missing or malformed
        # Use channel_count from stream info to generate generic labels
        num_channels = stream['info'].get('channel_count', ['0'])[0]
        try:
            num_channels = int(num_channels)
        except ValueError:
            num_channels = 1

        # Generate generic channel labels
        labels = [f"Channel {i}" for i in range(num_channels)]
        return labels, [''] * num_channels, [''] * num_channels


# ===========================================================
# String Normalization for Robust Matching
# ===========================================================

def normalize_string(s):
    """
    Normalize a string for robust matching by removing whitespace and case differences.

    This function is essential for handling copy-paste errors, where users might
    inadvertently include leading/trailing whitespace or use different casing
    when specifying stream or channel names.

    Parameters
    ----------
    s : str
        Input string to normalize

    Returns
    -------
    str
        Normalized string with leading/trailing whitespace removed and lowercase

    Notes
    -----
    - This function makes matching case-insensitive and whitespace-tolerant
    - Internal whitespace (between words) is preserved
    - Used internally by filter_stream_channels_robust() and select_streams_by_name()
    """
    return s.strip().lower()


# ===========================================================
# Channel Filtering with Robust Matching
# ===========================================================

def filter_stream_channels_robust(stream, channel_names):
    """
    Filter a stream to include only specified channels using robust name matching.

    This function creates a new stream containing only the channels whose names
    match those in the channel_names list. Matching is case-insensitive and
    whitespace-tolerant, making it resilient to copy-paste errors.

    Parameters
    ----------
    stream : dict
        Input stream dictionary containing:
        - 'info': Stream metadata
        - 'time_series': Data array of shape (n_samples, n_channels)
        - 'time_stamps': Optional timestamp array
    channel_names : list of str
        List of channel names to keep
        Names are matched using case-insensitive, whitespace-tolerant comparison

    Returns
    -------
    dict or None
        Filtered stream dictionary with the same structure as input, but containing
        only the matched channels. Returns None if no channels match.

    Notes
    -----
    - Original stream is not modified (creates a shallow copy of metadata)
    - Channel order in output matches the order of matches found, not input order
    - Duplicate channel names in channel_names are ignored (each channel appears once)
    - Metadata is updated to reflect the new channel count
    """
    # Extract all available channel information from the stream
    labels, types, units = extract_channel_labels(stream)

    # Create a mapping from normalized names to original indices
    # This allows O(1) lookup for each requested channel
    normalized_labels = {normalize_string(label): i for i, label in enumerate(labels)}

    # Normalize the requested channel names for comparison
    normalized_requests = [normalize_string(name) for name in channel_names]

    # Find indices of matching channels
    matching_indices = []
    matched_names = []
    for req in normalized_requests:
        if req in normalized_labels:
            idx = normalized_labels[req]
            matching_indices.append(idx)
            matched_names.append(labels[idx])  # Keep original label (not normalized)

    # Return None if no channels matched
    if not matching_indices:
        return None

    # Create filtered stream with only the selected channels
    filtered_stream = {
        'info': stream['info'].copy(),  # Shallow copy of metadata
        'time_series': stream['time_series'][:, matching_indices],  # Select columns
        'time_stamps': stream.get('time_stamps', None),  # Preserve timestamps
    }

    # Update metadata to reflect the filtered channel set
    try:
        desc = filtered_stream['info']['desc'][0]
        if 'channels' in desc and 'channel' in desc['channels'][0]:
            original_channels = desc['channels'][0]['channel']
            if isinstance(original_channels, list):
                # Keep only the channel metadata entries that matched
                filtered_channels = [original_channels[i] for i in matching_indices]
                filtered_stream['info']['desc'][0]['channels'][0]['channel'] = filtered_channels
        # Update the channel count in metadata
        filtered_stream['info']['channel_count'] = [str(len(matching_indices))]
    except Exception:
        # If metadata update fails, continue (data is still filtered correctly)
        pass

    return filtered_stream


# ===========================================================
# Multi-Stream Selection with Diagnostics
# ===========================================================

def select_streams_by_name(streams, selection_dict, verbose=True):
    """
    Select and filter multiple streams and their channels with robust name matching.

    This is the primary function for extracting desired data from a collection of
    XDF streams. It handles multiple streams simultaneously and provides detailed
    diagnostic output to help debug selection issues.

    Parameters
    ----------
    streams : list of dict
        List of stream dictionaries loaded from an XDF file
        Typically obtained via: streams, _ = pyxdf.load_xdf("file.xdf")
    selection_dict : dict
        Dictionary mapping stream names to lists of desired channel names.
        Both stream names and channel names use robust (case-insensitive,
        whitespace-tolerant) matching.
    verbose : bool, optional (default=True)
        If True, prints detailed diagnostic information

    Returns
    -------
    list of dict
        List of filtered stream dictionaries, each containing only the
        requested channels. Returns empty list if no streams match.

    Notes
    -----
    - Stream matching is case-insensitive and whitespace-tolerant
    - If a stream name doesn't match, it's skipped (not an error)
    - If none of a stream's requested channels match, that stream is skipped
    - Partial channel matches are allowed
    - Use verbose=True when debugging selection issues
    """
    selected_streams = []

    # Create a lookup table: normalized name -> (original stream, original name)
    stream_lookup = {}
    for stream in streams:
        stream_name = stream['info']['name'][0]
        normalized_name = normalize_string(stream_name)
        stream_lookup[normalized_name] = (stream, stream_name)

    # Print available streams if verbose mode is enabled
    if verbose:
        print(f"\n🔍 Available streams ({len(streams)}):")
        for norm_name, (_, orig_name) in stream_lookup.items():
            print(f"  '{orig_name}'")
        print()

    # Process each requested stream
    for requested_name, channels in selection_dict.items():
        normalized_request = normalize_string(requested_name)

        if "polar" in normalized_request:
            # Search for a stream with "Polar H10" in original name (case-insensitive)
            matching_stream = None
            for norm_name, (stream, orig_name) in stream_lookup.items():
                if "polar h10" in orig_name.lower():
                    matching_stream = (stream, orig_name)
                    break
            if matching_stream:
                stream, original_name = matching_stream
                if verbose:
                    print(f"✓ Found stream matching 'Polar H10': '{original_name}'")
                filtered_stream = filter_stream_channels_robust(stream, channels)

                if filtered_stream is not None:
                    labels, _, _ = extract_channel_labels(filtered_stream)
                    if verbose:
                        print(f"  Matched {len(labels)} channel(s): {labels}")
                    selected_streams.append(filtered_stream)
                else:
                    if verbose:
                        print(f"  No matching channels found in '{original_name}'")
                continue
        # Check if the stream exists in our lookup
        if normalized_request in stream_lookup:
            stream, original_name = stream_lookup[normalized_request]
            if verbose:
                print(f"✓ Found stream: '{original_name}'")
                print(f"  Searching for channels: {channels}")

            # Filter the stream to include only requested channels
            filtered_stream = filter_stream_channels_robust(stream, channels)

            if filtered_stream is not None:
                # Successfully found at least one matching channel
                labels, _, _ = extract_channel_labels(filtered_stream)
                if verbose:
                    print(f"  Matched {len(labels)} channel(s): {labels}")
                selected_streams.append(filtered_stream)
            else:
                # Stream was found, but none of the requested channels matched
                if verbose:
                    print(f"  No matching channels found in '{original_name}'")
        else:
            # Stream name didn't match any available streams
            if verbose:
                print(f"✗ Stream not found: '{requested_name}'")
                print(f"  (normalized as: '{normalized_request}')")

    # Print summary
    if verbose:
        print(f"\n📊 Total streams selected: {len(selected_streams)}\n")

    return selected_streams


# ===========================================================
# Stream-to-DataFrame Conversion with Synchronization
# ===========================================================

def streams_to_dataframe(streams, resample=True, target_freq=1.0, use_timestamps=True, n=0):
    """
    Convert multiple streams into a single synchronized pandas DataFrame.

    This function performs temporal alignment, resampling, and interpolation to
    combine multiple data streams with different sampling rates into a unified
    DataFrame with a common time base. This is essential for multi-modal analysis
    where different sensors record at different frequencies.

    Parameters
    ----------
    streams : list of dict
        List of stream dictionaries to combine
    resample : bool, optional (default=True)
        If True, resample all streams to a common frequency (target_freq)
    target_freq : float, optional (default=1.0)
        Target sampling frequency in Hz for resampling (only used if resample=True)
    use_timestamps : bool, optional (default=True)
        If True, use the original timestamps from each stream
    n : int, optional (default=0)
        Number of samples to truncate from the beginning AND end of the
        final DataFrame (total rows removed: 2*n)

    Returns
    -------
    pandas.DataFrame
        Synchronized DataFrame with columns:
        - 'Time': Common time base (in seconds)
        - '{StreamName}_{ChannelName}': One column per channel from each stream

    Notes
    -----
    - Empty input (streams=[]) returns an empty DataFrame
    - All streams must have some temporal overlap
    - Interpolation uses linear interpolation with extrapolation at boundaries
    - Channel naming follows the pattern: "{StreamName}_{ChannelLabel}"
    """
    # Handle empty input
    if not streams:
        print("⚠ No streams provided to streams_to_dataframe()")
        return pd.DataFrame()

    all_data = []

    # Process each stream individually
    for stream in streams:
        # Extract stream metadata
        name = stream['info']['name'][0]
        data = stream['time_series']
        ts = stream.get('time_stamps')
        labels, _, _ = extract_channel_labels(stream)

        # Safety check: ensure we have labels for all data columns
        num_channels = data.shape[1]
        labels = labels[:num_channels]

        # Determine time column
        if use_timestamps and ts is not None:
            time_col = ts  # Use original timestamps
        else:
            # Generate synthetic timestamps assuming constant sampling rate
            time_col = np.arange(len(data)) / target_freq

        # Create DataFrame for this stream with prefixed column names
        df_part = pd.DataFrame(data, columns=[f"{name}_{label}" for label in labels])
        df_part.insert(0, "Time", time_col)
        all_data.append(df_part)

    # Determine the overlapping time range across all streams
    # Use the latest start time and earliest end time
    min_t = max(df["Time"].min() for df in all_data)
    max_t = min(df["Time"].max() for df in all_data)

    # Create the new time grid
    if resample:
        # Uniform grid at target_freq Hz
        new_time = np.arange(min_t, max_t, 1.0 / target_freq)
    else:
        # Use the union of all original timestamps (sorted and unique)
        new_time = sorted(set(np.concatenate([df["Time"].values for df in all_data])))

    # Initialize the merged data dictionary
    merged = {"Time": new_time}

    # Interpolate each stream's channels onto the new time grid
    for df_part in all_data:
        for col in df_part.columns:
            if col == "Time":
                continue  # Skip the time column itself

            # Create interpolation function
            f = interp1d(df_part["Time"], df_part[col],
                        fill_value="extrapolate",
                        bounds_error=False)

            # Apply interpolation to new time grid
            merged[col] = f(new_time)

    # Create the final combined DataFrame
    df = pd.DataFrame(merged)

    # Truncate n samples from start and end if requested
    if n > 0:
        if len(df) > 2 * n:
            # Remove first n and last n rows
            df = df.iloc[n:-n].reset_index(drop=True)
        else:
            # Not enough data to truncate safely
            print(f"⚠ Warning: n={n} too large for dataset length {len(df)} — skipping truncation.")

    return df


# ===========================================================
# Post-Processing: Resampling
# ===========================================================

def resample_dataframe(data, target_freq=1.0, verbose=True):
    """
    Resample a DataFrame onto a uniform time grid using linear interpolation.

    This function takes an existing DataFrame with a 'Time' column and resamples
    all data columns to a specified frequency. Unlike streams_to_dataframe() which
    handles multiple streams, this function operates on an already-merged DataFrame
    and provides detailed diagnostics about time spacing.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing a 'Time' column and one or more data columns
    target_freq : float, optional (default=1.0)
        Desired sampling frequency in Hz (samples per second)
    verbose : bool, optional (default=True)
        If True, prints diagnostic information about time spacing and resampling

    Returns
    -------
    pandas.DataFrame
        Resampled DataFrame with uniform time grid and interpolated data values
    """
    # Validate input
    if "Time" not in data.columns:
        raise ValueError("Input DataFrame must contain a 'Time' column")

    # Extract original time column
    original_time = data["Time"].values

    if len(original_time) < 2:
        raise ValueError("DataFrame must have at least 2 time points for resampling")

    # Check original time spacing
    time_diffs = np.diff(original_time)
    tolerance = 1e-6
    is_equally_spaced = np.allclose(time_diffs, np.mean(time_diffs), atol=tolerance)

    if verbose:
        if is_equally_spaced:
            print("\n✓ Original time data is equally spaced.")
        else:
            print("\n⚠ Original time data is NOT equally spaced.")
        print(f"  Standard deviation of time differences: {np.std(time_diffs):.6f} seconds")

    # Create new uniform time grid
    new_time = np.arange(original_time[0], original_time[-1], 1.0 / target_freq)

    # Helper function for interpolating a single column
    def interpolate_column(original_time, col_data, new_time):
        """Interpolate a single data column onto a new time grid."""
        if len(original_time) < 2:
            return np.full(len(new_time), np.mean(col_data))
        else:
            interp_func = interp1d(
                original_time,
                col_data,
                kind='linear',
                fill_value="extrapolate",
                bounds_error=False
            )
            return interp_func(new_time)

    # Resample each data column
    columns_to_resample = [col for col in data.columns if col != "Time"]
    resampled_data = {}

    for col in columns_to_resample:
        col_data = data[col].values
        resampled_data[col] = interpolate_column(original_time, col_data, new_time)

    # Create final resampled DataFrame
    final_dataset = pd.DataFrame(resampled_data)
    final_dataset.insert(0, "Time", new_time)

    # Verify resampled time spacing
    resampled_time_diffs = np.diff(new_time)
    is_resampled_equally_spaced = np.allclose(
        resampled_time_diffs,
        np.mean(resampled_time_diffs),
        atol=tolerance
    )

    if verbose:
        if is_resampled_equally_spaced:
            print("\n✓ Resampled time data is equally spaced.")
        else:
            print("\n⚠ Resampled time data is NOT equally spaced.")
        print(f"  Standard deviation of resampled time differences: {np.std(resampled_time_diffs):.6e} seconds")

        print(f"\n📊 Dataset Summary:")
        print(f"  Shape: {final_dataset.shape} (rows × columns)")
        print(f"  Total data points: {final_dataset.size:,}")
        print(f"  Time range: {new_time[0]:.3f} - {new_time[-1]:.3f} seconds")
        print(f"  Duration: {new_time[-1] - new_time[0]:.3f} seconds")
        print(f"  Sampling rate: {target_freq} Hz")
        print(f"  Time step: {1.0/target_freq:.6f} seconds")

    return final_dataset


# ===========================================================
# Post-Processing: Truncation
# ===========================================================

def truncate_dataframe(data, n=0, verbose=True):
    """
    Remove n samples from the beginning and end of a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame to truncate
    n : int, optional (default=0)
        Number of rows to remove from BOTH the beginning and end
        Total rows removed = 2*n
    verbose : bool, optional (default=True)
        If True, prints truncation information

    Returns
    -------
    pandas.DataFrame
        Truncated DataFrame with first n and last n rows removed
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got n={n}")

    if n == 0:
        if verbose:
            print("\nℹ No truncation performed (n=0)")
        return data

    original_length = len(data)

    if original_length <= 2 * n:
        if verbose:
            print(f"\n⚠ Warning: n={n} is too large for dataset length {original_length} — skipping truncation.")
            print(f"   Minimum length required: {2*n + 1} (to remove {n} from each end)")
        return data

    if verbose:
        print(f"\n📊 Truncation Summary:")
        print(f"  Original shape: {data.shape}")
        print(f"  Removing {n} rows from start and end ({2*n} total)")

    truncated_data = data.iloc[n:-n].reset_index(drop=True)

    if verbose:
        print(f"  Final shape: {truncated_data.shape}")

        if "Time" in truncated_data.columns:
            time_removed_start = data.iloc[n]["Time"] - data.iloc[0]["Time"]
            time_removed_end = data.iloc[-1]["Time"] - data.iloc[-n-1]["Time"]
            print(f"  Time removed from start: {time_removed_start:.3f} seconds")
            print(f"  Time removed from end: {time_removed_end:.3f} seconds")
            print(f"  New time range: {truncated_data['Time'].iloc[0]:.3f} - {truncated_data['Time'].iloc[-1]:.3f} seconds")

    return truncated_data


# ===========================================================
# Complete Pipeline: All-in-One Processing
# ===========================================================

def process_xdf_streams(streams, selection_dict, target_freq=50.0, truncate_n=0, verbose=True):
    """
    Complete pipeline for processing XDF streams from selection to final DataFrame.

    This is the main high-level function that combines all processing steps:
    stream selection, synchronization, resampling, and truncation. Use this
    for a complete end-to-end workflow with a single function call.

    Parameters
    ----------
    streams : list of dict
        List of stream dictionaries from XDF file (from pyxdf.load_xdf())
    selection_dict : dict
        Dictionary mapping stream names to lists of channel names
        Uses robust (case-insensitive, whitespace-tolerant) matching
    target_freq : float, optional (default=50.0)
        Target sampling frequency in Hz for final resampled data
    truncate_n : int, optional (default=0)
        Number of samples to remove from beginning AND end of final DataFrame
        Total samples removed = 2 * truncate_n. Set to 0 to skip truncation
    verbose : bool, optional (default=True)
        If True, prints detailed progress information at each step

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with synchronized, resampled, and optionally truncated data

    Processing Pipeline
    -------------------
    1. Stream Selection - Filter streams and channels
    2. Synchronization - Align all streams to common time range
    3. Resampling - Interpolate to uniform target_freq
    4. Truncation - Remove truncate_n samples from start and end
    """
    if verbose:
        print("\n" + "="*70)
        print("XDF STREAM PROCESSING PIPELINE")
        print("="*70)
        print("\n📋 Configuration:")
        print(f"  Target frequency: {target_freq} Hz")
        print(f"  Truncation: {truncate_n} samples from each end")
        print(f"  Streams requested: {len(selection_dict)}")

    # STEP 1: Stream Selection
    if verbose:
        print("\n" + "-"*70)
        print("STEP 1: Stream Selection")
        print("-"*70)

    selected_streams = select_streams_by_name(streams, selection_dict, verbose=verbose)

    if not selected_streams:
        if verbose:
            print("\n❌ No streams selected. Check your selection_dict.")
        return pd.DataFrame()

    # STEP 2: Synchronization and Initial Processing
    if verbose:
        print("\n" + "-"*70)
        print("STEP 2: Synchronization and Resampling")
        print("-"*70)

    df = streams_to_dataframe(
        selected_streams,
        resample=True,
        target_freq=target_freq,
        use_timestamps=True,
        n=0
    )

    if df.empty:
        if verbose:
            print("\n❌ Empty DataFrame after synchronization.")
        return df

    if verbose:
        print(f"\n✓ Synchronized DataFrame created")
        print(f"  Shape: {df.shape}")
        print(f"  Time range: {df['Time'].iloc[0]:.3f} - {df['Time'].iloc[-1]:.3f} seconds")
        print(f"  Duration: {df['Time'].iloc[-1] - df['Time'].iloc[0]:.3f} seconds")

    # STEP 3: Final Resampling
    if verbose:
        print("\n" + "-"*70)
        print("STEP 3: Final Resampling Verification")
        print("-"*70)

    df_resampled = resample_dataframe(df, target_freq=target_freq, verbose=verbose)

    # STEP 4: Truncation
    if truncate_n > 0:
        if verbose:
            print("\n" + "-"*70)
            print("STEP 4: Edge Truncation")
            print("-"*70)

        df_final = truncate_dataframe(df_resampled, n=truncate_n, verbose=verbose)
    else:
        df_final = df_resampled
        if verbose:
            print("\n" + "-"*70)
            print("STEP 4: Edge Truncation")
            print("-"*70)
            print("\nℹ Truncation skipped (truncate_n=0)")

    # Final Summary
    if verbose:
        print("\n" + "="*70)
        print("PROCESSING COMPLETE ✓")
        print("="*70)
        print(f"\n📊 Final Dataset Summary:")
        print(f"  Shape: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")
        print(f"  Time range: {df_final['Time'].iloc[0]:.3f} - {df_final['Time'].iloc[-1]:.3f} seconds")
        print(f"  Duration: {df_final['Time'].iloc[-1] - df_final['Time'].iloc[0]:.3f} seconds")
        print(f"  Sampling frequency: {target_freq} Hz")
        print(f"  Columns: {list(df_final.columns)}")
        print("\n" + "="*70 + "\n")

    return df_final


# ===========================================================
# Diagnostic and Utility Functions
# ===========================================================

def print_stream_info(streams):
    """
    Print detailed, formatted information about available streams and channels.

    This function provides a human-readable overview of all streams in a
    collection, showing their names, channel counts, and individual channel
    details (labels, types, units). Useful for exploring unknown XDF files
    and deciding which streams/channels to extract.

    Parameters
    ----------
    streams : list of dict
        List of stream dictionaries (typically from pyxdf.load_xdf())
    """
    print(f"\n{'='*60}")
    print(f"Found {len(streams)} stream(s):")
    print(f"{'='*60}\n")

    for stream in streams:
        name = stream['info']['name'][0]
        labels, types, units = extract_channel_labels(stream)

        print(f"📡 Stream: {name}")
        print(f"   Channels ({len(labels)}):")

        for l, t, u in zip(labels, types, units):
            type_str = f"({t})" if t else ""
            unit_str = f"[{u}]" if u else ""
            print(f"      • {l} {type_str} {unit_str}".strip())

        print(f"{'-'*60}\n")


def get_copyable_format(streams):
    """
    Generate a copyable Python dictionary template for stream selection.

    This utility function creates a ready-to-use selection_dict template
    that includes all available streams and their channels. Users can
    copy this output and simply delete unwanted streams/channels rather
    than typing names manually (which reduces typos).

    Parameters
    ----------
    streams : list of dict
        List of stream dictionaries (typically from pyxdf.load_xdf())
    """
    print("\n📋 Copy this template and fill in your desired channels:\n")
    print("selection_dict = {")

    for stream in streams:
        name = stream['info']['name'][0]
        labels, _, _ = extract_channel_labels(stream)
        print(f'    "{name}": {labels},')

    print("}\n")


if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")
    print("See example_usage.py for usage examples.")