# Transit Detector Web App - PRODUCTION

## Quick Start
```bash
python app_v8.py
```
Access at: http://127.0.0.1:5000

## Available Models
- **CNN** (Default): Best performance, 100% detection rate
- **ResNet**: Alternative with lower false positives  
- **CNN+ResNet Ensemble**: Combined models

## Features
- Telescope data download (Kepler, TESS)
- CSV file upload support
- Real-time transit detection
- Visualization of results
- Multiple model architectures

## Model Performance
| Model | Detection Rate | False Positives | Status |
|-------|---------------|----------------|--------|
| CNN | 100% | High | ✅ Recommended |
| ResNet | 50% | Medium | ⚠️ Alternative |
| Ensemble | 100% | Medium | ✅ Balanced |

## Files
- `app_v8.py` - Main Flask application
- `models/` - Trained model files
- `templates/` - HTML templates
- `static/uploads/` - File upload directory

## Requirements
```bash
pip install -r requirements.txt
```

## Troubleshooting
- For high false positives: Try ResNet model
- Make sure models folder has cnn_best.pt and resnet_epoch29.pt

Last Updated: Production Ready