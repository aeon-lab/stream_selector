# XDF Stream Selector

A Python toolkit for extracting, filtering, synchronizing, and resampling multi-modal data streams from XDF (Extensible Data Format) files, plus utilities for survey data cleanup.

## Features

### XDF Processing
- 🔍 **Robust name matching** - Case-insensitive and whitespace-tolerant stream/channel selection
- 🔄 **Multi-stream synchronization** - Align streams with different sampling rates
- ⚡ **Flexible resampling** - Interpolate data to any target frequency
- ✂️ **Edge truncation** - Remove unstable data from recording boundaries
- 🛠️ **Diagnostic tools** - Debug stream selection issues easily
- 📊 **All-in-one pipeline** - Process everything with a single function call

### Survey Data Processing
- 🧹 **Automatic cleanup** - Remove invalid entries and format inconsistencies
- 🔢 **Smart type conversion** - Handle numeric fields with commas and special characters
- 📅 **Year-to-experience conversion** - Automatically convert birth years to experience values
- ✅ **Data validation** - Filter out non-numeric participant IDs

## Installation

```bash
pip install pandas numpy scipy pyxdf
```

Then download `xdf_stream_processing.py` and place it in your project directory.

## Quick Start

### XDF Processing

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

## Survey Data Processing

The toolkit includes utilities for cleaning survey data, particularly useful for preprocessing questionnaire responses before analysis.

### Features

- **Participant ID cleanup**: Remove special characters (e.g., '#5' → '5')
- **Invalid entry filtering**: Remove non-numeric participant IDs (e.g., rows starting with letters)
- **Numeric field cleaning**: Remove commas from numeric fields (e.g., '1,000' → '1000')
- **Smart year conversion**: Convert year values to experience (e.g., '1995' → '30' for experience)

### Usage Example

```python
import pandas as pd

# Load your survey data
survey_data = pd.read_csv("raw_survey.csv")

# Make a proper copy to avoid warnings
selected_data = survey_data.copy()

# Clean participant IDs - remove '#' prefix
selected_data['pid'] = selected_data['pid'].astype(str).str.replace('#', '', regex=False)

# Filter out invalid participant IDs (keep only numeric)
selected_data = selected_data[selected_data['pid'].str.match(r'^\d', na=False)].copy()

# Remove commas from numeric fields (e.g., flight hours)
selected_data['flt_hrs'] = selected_data['flt_hrs'].astype(str).str.replace(',', '', regex=False)

# Clean experience column with smart year detection
def clean_experience(val):
    try:
        num = float(val.replace(',', ''))
        if num >= 1900:  # Likely a year, convert to experience
            return str(int(2025 - num))
        else:  # Already an experience value
            return str(int(num))
    except:
        return val  # Keep original if conversion fails

selected_data['experience'] = selected_data['experience'].astype(str).apply(clean_experience)

# Save cleaned data
clean_survey_data = selected_data.copy()
clean_survey_data.to_csv('clean_survey_data.csv', index=False)

print(f"Cleaned {len(clean_survey_data)} survey responses")
```

### Common Survey Cleanup Steps

1. **Remove special characters from IDs**
   ```python
   df['pid'] = df['pid'].astype(str).str.replace('#', '', regex=False)
   ```

2. **Filter invalid entries**
   ```python
   df = df[df['pid'].str.match(r'^\d', na=False)].copy()
   ```

3. **Clean numeric fields**
   ```python
   df['numeric_field'] = df['numeric_field'].astype(str).str.replace(',', '', regex=False)
   ```

4. **Convert years to age/experience**
   ```python
   def year_to_experience(val, current_year=2025):
       try:
           num = float(val.replace(',', ''))
           if num >= 1900:
               return str(int(current_year - num))
           return str(int(num))
       except:
           return val
   
   df['experience'] = df['experience'].astype(str).apply(year_to_experience)
   ```

### Validation Checks

After cleaning, always verify your data:

```python
print(f"Shape after cleaning: {clean_survey_data.shape}")
print(f"\nUnique participant IDs: {clean_survey_data['pid'].nunique()}")
print(f"\nCleaned experience values:\n{clean_survey_data['experience'].value_counts()}")
```

## Key Parameters

### XDF Processing

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

### XDF Streams

#### High-frequency signals (ECG, EEG)
```python
df = process_xdf_streams(streams, selection, target_freq=250.0)
```

#### Low-frequency signals (HR, pupil diameter)
```python
df = process_xdf_streams(streams, selection, target_freq=10.0)
```

#### No resampling (keep original rates)
```python
df = streams_to_dataframe(streams, resample=False)
```

#### Remove edge effects
```python
df = process_xdf_streams(streams, selection, truncate_n=200)
```

### Survey Data

#### Clean pilot study data
```python
# Remove test entries and format fields
survey = survey[survey['pid'].str.match(r'^\d', na=False)].copy()
survey['flt_hrs'] = survey['flt_hrs'].str.replace(',', '')
```

#### Prepare for statistical analysis
```python
# Convert all numeric fields to proper types
numeric_cols = ['flt_hrs', 'experience', 'age']
for col in numeric_cols:
    survey[col] = pd.to_numeric(survey[col], errors='coerce')
```

## Examples

See `example_usage.py` for comprehensive examples including:
1. Basic workflow with synthetic data
2. Loading and processing real XDF files
3. Custom processing workflows
4. Diagnostic and debugging tools
5. Multiple processing frequencies
6. Survey data cleanup workflows

Run examples:
```bash
python example_usage.py
```

## Functions Overview

### XDF Processing

| Function | Purpose |
|----------|---------|
| `process_xdf_streams()` | Complete pipeline (recommended) |
| `print_stream_info()` | Display available streams/channels |
| `get_copyable_format()` | Generate selection template |
| `select_streams_by_name()` | Filter streams and channels |
| `streams_to_dataframe()` | Synchronize and merge streams |
| `resample_dataframe()` | Resample to uniform frequency |
| `truncate_dataframe()` | Remove edge samples |

### Survey Processing

Survey data cleanup functions are provided as code snippets in the documentation above. For a complete implementation, see the examples directory.

## Requirements

- Python 3.7+
- pandas
- numpy
- scipy
- pyxdf (for loading XDF files)

## Citation

If you use XDF Stream Selector in your research, please cite:

```bibtex
@software{xdf_stream_selector,
  author = {Md Mijanur Rahman, Niklas P. Schulmeyer},
  title = {XDF Stream Selector: A Python Toolkit for Multi-Modal Data Stream Processing},
  year = {2025},
  url = {https://github.com/aeon-lab/stream_selector},
  version = {1.0.0}
}
```

## License

This project is licensed under the MIT License - see below for details.

```
MIT License

Copyright (c) 2025 [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Authors

Md Mijanur Rahman  
Niklas P. Schulmeyer

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.