Cosmic Classifier: An End-to-End Deep Learning Pipeline for Exoplanet Detection
Cosmic Classifier is a complete, automated system for discovering transiting exoplanets in stellar light curve data. It leverages a multi-stage data processing pipeline and an ensemble of deep learning models to deliver high-accuracy predictions through a user-friendly web interface.

Key Features
End-to-End Automation: Automates the entire workflow from data acquisition on NASA's archives to a final, deployable prediction engine.

Scalable Data Handling: Built to process terabytes of data using high-performance multiprocessing, memory management, and a robust data storage strategy with Google Drive.

High-Accuracy Ensemble Model: Employs a sophisticated ensemble of CNN and Attention models, ensuring more reliable predictions than any single model.

Interactive Web Application: A simple Flask web app allows users to analyze official telescope data or upload their own light curves for instant analysis.

Robust Inference Engine: Features a hybrid detection mode that uses a precise Box-Least-Squares (BLS) algorithm with an automatic fallback to a comprehensive sliding-window analysis.

Reproducible Workflow: The entire pipeline is scripted, modular, and designed for full reproducibility.

How It Works: The Cosmic Classifier Pipeline
The project is structured as a four-phase pipeline, where the output of each stage becomes the input for the next. This modular design makes the system easy to manage, test, and upgrade.

Phase 1: Data Acquisition & Storage

complete_fetcher_v15_3_original.py

Queries the NASA Exoplanet Archive for confirmed planets (positives) and known false positives (negatives).

Downloads full light curves from Kepler, K2, and TESS missions.

Efficiently batches, zips, and uploads data to Google Drive for persistent storage.

Phase 2: Preprocessing & Feature Engineering

exoplanet_master_csv_preprocessor_v15.py: Cleans and normalizes raw light curves into relative flux, merging all data into master CSV files split by label (planet vs. non-planet).

BLS_Transit_Window_Extractor_V13.py: Uses the Box-Least-Squares (BLS) algorithm to find transit signals. It then extracts millions of small, 256-point "windows" centered on these signals, transforming the time-series problem into an image-like classification task.

Phase 3: Model Training

Transit_Window_Trainer_V3.py

Trains three distinct deep learning architectures (CNN, ResNet, Attention) on the extracted windows.

Rigorously compares model performance and saves the best checkpoints.

Uses a balanced sampler to ensure models train effectively on the rare transit events.

Phase 4: Deployment & Inference

transit_detector_app_fixed.py

A Flask-based web application provides the user interface for real-time analysis.

It loads the trained models and uses an ensemble prediction strategy for the final verdict.

The Cosmic Classifier Web Application
The final output is a simple, powerful tool for exoplanet detection.

Features:
Live Telescope Data Analysis: Enter a star's ID (e.g., KIC 10593626) to download and analyze its data directly.

Custom Data Upload: Upload your own light curve in a CSV format for analysis.

Clear Verdicts & Visuals: Get a straightforward "TRANSIT DETECTED" or "NO TRANSIT" verdict, along with plots showing the model's confidence.

Setup and Installation
Prerequisites
Python 3.9+

pip package installer

Git

1. Clone the Repository
git clone [https://github.com/your-username/Cosmic-Classifier.git](https://github.com/your-username/Cosmic-Classifier.git)
cd Cosmic-Classifier

2. Set Up a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install Dependencies
A requirements.txt file can be created from the imported libraries. For now, install the core packages:

pip install torch pandas numpy lightkurve astroquery flask flask-cors matplotlib psutil tqdm

4. Google Drive API Setup (for Data Acquisition)
To run the data fetching script, you need to authenticate with Google Drive.

Follow the Google Drive API Python Quickstart to enable the API and download a credentials.json file.

Place the credentials.json file in the root directory of this project.

The first time you run complete_fetcher_v15_3_original.py, you will be prompted to authorize the application. A token.json file will be created to store your credentials for future runs.

How to Run the Full Pipeline
Follow these steps in order to reproduce the entire project, from data collection to a trained, runnable model.

Step 1: Data Acquisition
This script will download light curves from NASA and upload them to your Google Drive. It can take many hours to run completely.

python complete_fetcher_v15_3_original.py --resume

After this step, you will have a data/ directory containing subfolders with the downloaded light curve CSVs.

Step 2: Preprocess Raw Data
This merges all downloaded data into two master CSV files, normalized and ready for feature extraction.

python exoplanet_master_csv_preprocessor_v15.py

This will create exoplanet_master_dataset_v15_label_1.csv (non-planets) and exoplanet_master_dataset_v15_label_2.csv (planets).

Step 3: Extract Training Windows with BLS
This is a computationally intensive step that runs BLS and extracts millions of training windows.

# Optimized for your system, e.g., 16 cores
python BLS_Transit_Window_Extractor_V13.py --all --workers 16

This will create an extracted_windows_safe/ directory containing the .npy and metadata files for training.

Step 4: Train the Models
This script trains the CNN, ResNet, and Attention models and generates a performance report. This requires a CUDA-enabled GPU for reasonable training times.

python Transit_Window_Trainer_V3.py --all

This will save the best model checkpoints (e.g., cnn-epoch-XX-val_spec-X.XX.ckpt) inside checkpoints_multi_arch/.

Step 5: Convert Checkpoints for Inference
Convert the saved training checkpoints into lightweight, portable .pt files for the web app.

# Example for the best CNN model
python convert_checkpoint_to_pt.py checkpoints_multi_arch/cnn/cnn-best-model.ckpt -o models/cnn_best.pt -a cnn

# Example for the best Attention model #not working at the moment
python convert_checkpoint_to_pt.py checkpoints_multi_arch/attention/attention-best-model.ckpt -o models/attention_best.pt -a attention

Running the Web Application
Once you have your converted .pt model files, you can run the web application.

Place Models: Make sure your cnn_best.pt and attention_best.pt files are inside a models/ directory in the project root.

Start the Server:

python transit_detector_app_fixed.py

Open in Browser: Open your web browser and navigate to http://127.0.0.1:5000.

You can now use the application to detect exoplanets!

Attention modell is not working at the moment
