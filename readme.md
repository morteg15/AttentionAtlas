# Algae Prediction Visualizer

This project uses machine learning to predict harmful algal blooms based on oceanographic data. It includes functionality for data loading, model training, and visualization of predictions.

We have also included a Demo web app found in the demo_web_app folder this can be runn by opening the index.html file to show how a future application could look like

## Features

- Loads and processes NetCDF oceanographic data and CSV event data
- Trains a Random Forest Classifier to predict harmful algal blooms
- Visualizes predictions on a map using Cartopy
- Supports saving and loading trained models

## Requirements

See `requirements.txt` for a list of required Python packages.

dataset needs to be downloaded
https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_BGC_001_029/description
and	https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/

## Usage

1. Ensure you have the required data files:

   - NetCDF file with oceanographic data
   - CSV file with harmful algal bloom event data
2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```
3. Run the main script:

   ```
   python algae_prediction_visualizer.py
   ```

## File Structure

- `algae_prediction_visualizer.py`: Main script containing the `AlgaePredictionVisualizer` class
- `data/`: Directory containing input data files
  - `cmems_mod_glo_bgc_my_0.25deg_P1D-m_1726265418849.nc`: NetCDF file with oceanographic data
  - `haedat_search.csv`: CSV file with harmful algal bloom event data
- `requirements.txt`: List of required Python packages
- `README.md`: This file

## Output

The script generates:

- A trained model file (`algae_model.joblib`)
- A visualization of a sample prediction from 2019 (`sample_prediction_YYYYMMDD.png`)

## License

[Specify your license here]

## Contact

[Your contact information or project repository link]
