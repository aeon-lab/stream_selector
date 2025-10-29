# XDF Stream Selector

A Python toolkit for extracting, filtering, synchronizing, and resampling multi-modal data streams from XDF (Extensible Data Format) files.

## Features

- 🔍 **Robust name matching** - Case-insensitive and whitespace-tolerant stream/channel selection
- 🔄 **Multi-stream synchronization** - Align streams with different sampling rates
- ⚡ **Flexible resampling** - Interpolate data to any target frequency
- ✂️ **Edge truncation** - Remove unstable data from recording boundaries
- 🛠️ **Diagnostic tools** - Debug stream selection issues easily
- 📊 **All-in-one pipeline** - Process everything with a single function call

## Installation

```bash
pip install pandas numpy scipy pyxdf
```

Then download `xdf_stream_processing.py` and place it in your project directory.

## Quick Start

```python
import pyxdf
from xdf_stream_processing import process_xdf_streams, print_stream_info

# Load your XDF file
streams, header = pyxdf.load_xdf("recording.xdf")

# Inspect available streams
print_stream_info(streams)

# Define what you want to extract
selection = {
    "VarjoEyeMetrics": ["leftPupilDiam", "rightPupilDiam"],
    "Polar H10": ["HR", "RRI"],
}

# Process everything in one call
df = process_xdf_streams(
    streams=streams,
    selection_dict=selection,
    target_freq=50.0,    # Resample to 50 Hz
    truncate_n=100,      # Remove 100 samples from each end
    verbose=True
)

# Save your processed data
df.to_csv("processed_data.csv", index=False)
```

## Usage

### 1. Inspect Your Data

```python
from xdf_stream_processing import print_stream_info, get_copyable_format

# See all available streams and channels
print_stream_info(streams)

# Get a copyable template for selection_dict
get_copyable_format(streams)
```

### 2. Select and Process Streams

The main function handles everything:

```python
df = process_xdf_streams(
    streams=streams,              # From pyxdf.load_xdf()
    selection_dict=selection,     # What to extract
    target_freq=50.0,            # Target sampling rate (Hz)
    truncate_n=100,              # Edge samples to remove
    verbose=True                 # Show progress
)
```

### 3. Custom Workflows

For more control, use individual functions:

```python
from xdf_stream_processing import (
    select_streams_by_name,
    streams_to_dataframe,
    resample_dataframe,
    truncate_dataframe
)

# Step-by-step processing
selected = select_streams_by_name(streams, selection)
df = streams_to_dataframe(selected, target_freq=50.0)
df = resample_dataframe(df, target_freq=100.0)
df = truncate_dataframe(df, n=50)
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `target_freq` | Target sampling frequency in Hz | 50.0 |
| `truncate_n` | Samples to remove from each end | 0 |
| `verbose` | Print detailed progress information | True |
| `use_timestamps` | Use original timestamps vs synthetic | True |

## Selection Dictionary Format

The `selection_dict` maps stream names to lists of channel names:

```python
selection_dict = {
    "StreamName1": ["channel1", "channel2", "channel3"],
    "StreamName2": ["channelA", "channelB"],
    # Add more streams as needed
}
```

**Features:**
- ✅ Case-insensitive matching
- ✅ Whitespace-tolerant
- ✅ Partial matches allowed (extracts available channels)
- ✅ Special handling for Polar H10 devices

## Output Format

The processed DataFrame contains:
- `Time` column: Common time base in seconds
- `{StreamName}_{ChannelName}` columns: One per selected channel

Example:
```
     Time  VarjoEyeMetrics_leftPupilDiam  VarjoEyeMetrics_rightPupilDiam  Polar H10_HR
0   0.000                          3.245                           3.156         72.3
1   0.020                          3.248                           3.159         72.3
2   0.040                          3.251                           3.162         72.4
...
```

## Common Use Cases

### High-frequency signals (ECG, EEG)
```python
df = process_xdf_streams(streams, selection, target_freq=250.0)
```

### Low-frequency signals (HR, pupil diameter)
```python
df = process_xdf_streams(streams, selection, target_freq=10.0)
```

### No resampling (keep original rates)
```python
df = streams_to_dataframe(streams, resample=False)
```

### Remove edge effects
```python
df = process_xdf_streams(streams, selection, truncate_n=200)
```

## Examples

See `example_usage.py` for comprehensive examples including:
1. Basic workflow with synthetic data
2. Loading and processing real XDF files
3. Custom processing workflows
4. Diagnostic and debugging tools
5. Multiple processing frequencies

Run examples:
```bash
python example_usage.py
```

## Functions Overview

| Function | Purpose |
|----------|---------|
| `process_xdf_streams()` | Complete pipeline (recommended) |
| `print_stream_info()` | Display available streams/channels |
| `get_copyable_format()` | Generate selection template |
| `select_streams_by_name()` | Filter streams and channels |
| `streams_to_dataframe()` | Synchronize and merge streams |
| `resample_dataframe()` | Resample to uniform frequency |
| `truncate_dataframe()` | Remove edge samples |

## Requirements

- Python 3.7+
- pandas
- numpy
- scipy
- pyxdf (for loading XDF files)

## License

[Add your license here]

## Author

[Your Name]

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
