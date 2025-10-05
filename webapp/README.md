# Transit Detector Web Application

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Your Model

Copy your trained `.pt` model file to the `models/` folder:

```bash
# Example
cp /path/to/your/cnn_best.pt models/
```

### 3. Run the Application

```bash
python transit_detector_app_fixed.py
```

Open your browser to: **http://localhost:5000**

## Usage

### Option 1: Download from Telescope

1. Select "🌐 Download from Telescope"
2. Upload your `.pt` model file
3. Enter target ID (e.g., `KIC 10593626`)
4. Select mission (Kepler, TESS, K2)
5. Click "🔍 Detect Transits"

### Option 2: Upload CSV File

1. Select "📁 Upload CSV File"
2. Upload your `.pt` model file
3. Upload CSV lightcurve (1 row = 1 lightcurve)
4. (Optional) Enter row number or planet ID column
5. Select data normalization status
6. Click "🔍 Detect Transits"

## CSV Format

```csv
planet_id,flux_0,flux_1,flux_2,...,flux_N
KIC_12345,0.9988,0.9992,0.9985,...,0.9991
```

**Rules:**
- 1 row = 1 full lightcurve
- Columns named `flux_0`, `flux_1`, etc. OR all numeric
- Optional ID column
- NaN values auto-removed

## Features

✅ **Auto-detection**: Model architecture and settings loaded from .pt file
✅ **Hybrid BLS mode**: Automatic fallback to sliding window
✅ **High-confidence verdict**: Detects sparse transits correctly
✅ **GPU support**: Automatic GPU detection and usage
✅ **Dual input**: Telescope download OR CSV upload

## Model Files

Place your converted `.pt` models in the `models/` directory. To convert from `.ckpt`:

```bash
cd ../utils
python convert_checkpoint_to_pt.py /path/to/checkpoint.ckpt -a cnn -o ../webapp/models/cnn.pt
```

## Directory Structure

```
webapp/
├── transit_detector_app_fixed.py  # Main Flask application
├── templates/
│   └── transit_detector.html      # Web interface
├── static/
│   └── uploads/                   # Temporary upload folder
├── models/                        # Place your .pt models here
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Configuration

The app runs on `localhost:5000` by default. To change:

Edit the last line of `transit_detector_app_fixed.py`:

```python
app.run(host='0.0.0.0', port=8080, debug=False)  # Change as needed
```

## Troubleshooting

### "No module named 'flask'"
```bash
pip install flask flask-cors
```

### "No module named 'lightkurve'"
```bash
pip install lightkurve
```

### "CUDA out of memory"
The app will automatically fall back to CPU if GPU memory is insufficient.

### "Only .pt files accepted"
Convert your checkpoint first using the converter in `../utils/convert_checkpoint_to_pt.py`

## Documentation

See the `../docs/` folder for complete documentation:
- `CSV_UPLOAD_GUIDE.md` - Detailed CSV upload guide
- `STATUS_v2.md` - System capabilities
- `FINAL_UPDATE_SUMMARY.md` - Latest updates

## Performance

- **Telescope mode**: 10-30 seconds (depends on download)
- **CSV mode**: 5-15 seconds (local processing)
- **GPU**: ~5x faster than CPU for large lightcurves

## Tested On

✅ KIC 10593626 (Kepler-3b): DETECTED (91% detection rate)
✅ KIC 8462852 (Tabby's Star): NO TRANSIT (0% detection rate)
✅ Master CSV files (label_1 and label_2)
✅ Custom synthetic lightcurves
