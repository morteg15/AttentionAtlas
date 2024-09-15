import xarray as xr
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class AlgaePredictionVisualizer:
    def __init__(self, netcdf_file_path, norway_file_path):
        self.netcdf_file_path = netcdf_file_path
        self.norway_file_path = norway_file_path
        self.netcdf_ds = None
        self.norway_df = None
        self.model = None
        self.full_df = None

    def load_data(self):
        print("Loading NetCDF data...")
        self.netcdf_ds = xr.open_dataset(self.netcdf_file_path)
        print("\nLoading Norway event data...")
        self.norway_df = pd.read_csv(self.norway_file_path, encoding="ISO-8859-1")
        
        # Convert date columns to datetime
        date_columns = ['eventDate', 'initialDate', 'finalDate']
        for col in date_columns:
            self.norway_df[col] = pd.to_datetime(self.norway_df[col], errors='coerce')

    def match_events_with_data(self):
        print("\nMatching events with NetCDF data...")
        matched_data = []
        for _, event in self.norway_df.iterrows():
            event_date = event['eventDate']
            if pd.isnull(event_date):
                event_date = event['initialDate']
            if pd.isnull(event_date):
                event_date = event['finalDate']
            
            if pd.isnull(event_date):
                continue

            time_index = self.netcdf_ds.time.sel(time=event_date, method='nearest')
            
            if pd.notnull(event['latitude']) and pd.notnull(event['longitude']):
                lat_index = self.netcdf_ds.latitude.sel(latitude=event['latitude'], method='nearest')
                lon_index = self.netcdf_ds.longitude.sel(longitude=event['longitude'], method='nearest')
            else:
                lat_index = random.choice(self.netcdf_ds.latitude.values)
                lon_index = random.choice(self.netcdf_ds.longitude.values)
            
            try:
                chlorophyll = self.netcdf_ds.chl.sel(time=time_index, latitude=lat_index, longitude=lon_index).values.item()
                nitrate = self.netcdf_ds.no3.sel(time=time_index, latitude=lat_index, longitude=lon_index).values.item()
                phosphate = self.netcdf_ds.po4.sel(time=time_index, latitude=lat_index, longitude=lon_index).values.item()
                
                month = pd.to_datetime(time_index.values).month
                
                matched_data.append({
                    'time': time_index.values,
                    'latitude': lat_index,
                    'longitude': lon_index,
                    'month': month,
                    'chlorophyll': chlorophyll,
                    'nitrate': nitrate,
                    'phosphate': phosphate,
                    'is_harmful': 1
                })
            except (KeyError, ValueError) as e:
                print(f"Error processing event: {e}")
                continue

        return pd.DataFrame(matched_data)

    def generate_negative_examples(self, num_samples, existing_events):
        print("\nGenerating negative examples...")
        negative_samples = []
        
        while len(negative_samples) < num_samples:
            random_time = random.choice(self.netcdf_ds.time.values)
            random_lat = random.choice(self.netcdf_ds.latitude.values)
            random_lon = random.choice(self.netcdf_ds.longitude.values)
            
            if not any((existing_events['time'] == random_time) & 
                       (existing_events['latitude'] == random_lat) & 
                       (existing_events['longitude'] == random_lon)):
                try:
                    chlorophyll = self.netcdf_ds.chl.sel(time=random_time, latitude=random_lat, longitude=random_lon).values.item()
                    nitrate = self.netcdf_ds.no3.sel(time=random_time, latitude=random_lat, longitude=random_lon).values.item()
                    phosphate = self.netcdf_ds.po4.sel(time=random_time, latitude=random_lat, longitude=random_lon).values.item()
                    
                    month = pd.to_datetime(random_time).month
                    
                    negative_samples.append({
                        'time': random_time,
                        'latitude': random_lat,
                        'longitude': random_lon,
                        'month': month,
                        'chlorophyll': chlorophyll,
                        'nitrate': nitrate,
                        'phosphate': phosphate,
                        'is_harmful': 0
                    })
                except (KeyError, ValueError) as e:
                    print(f"Error generating negative sample: {e}")
                    continue
        
        return pd.DataFrame(negative_samples)

    def prepare_data(self):
        matched_df = self.match_events_with_data()
        negative_df = self.generate_negative_examples(len(matched_df), matched_df)
        self.full_df = pd.concat([matched_df, negative_df], ignore_index=True)
        
        X = self.full_df[['month', 'chlorophyll', 'nitrate', 'phosphate']]
        y = self.full_df['is_harmful']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        print("\nTraining the model...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        # Print feature importances
        feature_importance = self.model.feature_importances_
        for feature, importance in zip(X_train.columns, feature_importance):
            print(f"Feature: {feature}, Importance: {importance}")

    def save_model(self, filename='algae_model.joblib'):
        joblib.dump(self.model, filename)
        print(f"\nModel saved as {filename}")

    def load_model(self, filename='algae_model.joblib'):
        self.model = joblib.load(filename)
        print(f"\nModel loaded from {filename}")

    def predict_and_visualize_sample(self, sample):
        if self.model is None:
            print("Error: Model not trained or loaded.")
            return

        # Make prediction
        features = sample[['month', 'chlorophyll', 'nitrate', 'phosphate']].values.reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Calculate confidence for the predicted class
        confidence = probabilities[prediction] * 100  # Convert to percentage

        # Visualize
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': proj})

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)

        # Set the extent based on data
        lon_min, lon_max = self.netcdf_ds.longitude.min().item(), self.netcdf_ds.longitude.max().item()
        lat_min, lat_max = self.netcdf_ds.latitude.min().item(), self.netcdf_ds.latitude.max().item()
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

        # Plot chlorophyll data
        time_index = np.where(self.netcdf_ds.time.values == sample['time'])[0][0]
        chlorophyll_data = self.netcdf_ds.chl.isel(time=time_index).values
        
        # Check the shape of chlorophyll_data and adjust accordingly
        if len(chlorophyll_data.shape) == 3:
            chlorophyll_data = chlorophyll_data[0]  # Take the first slice if it's 3D
        elif len(chlorophyll_data.shape) > 3:
            print(f"Unexpected chlorophyll data shape: {chlorophyll_data.shape}")
            return

        # Create a masked array to handle NaN values
        masked_chlorophyll = np.ma.masked_invalid(chlorophyll_data)

        im = ax.imshow(masked_chlorophyll, extent=[lon_min, lon_max, lat_min, lat_max],
                       transform=proj, cmap='viridis', origin='lower')
        plt.colorbar(im, label='Chlorophyll')

        # Plot the sample location
        ax.scatter(sample['longitude'], sample['latitude'], 
                   color='red', s=100, transform=proj, marker='*', 
                   label=f'Sample location')

        # Set title with sample information and prediction
        plt.title(f"Chlorophyll on {pd.to_datetime(sample['time']).strftime('%Y-%m-%d')}\n"
                  f"Predicted: {'Harmful' if prediction else 'Not Harmful'} "
                  f"(Confidence: {confidence:.1f}%)\n"
                  f"Actual: {'Harmful' if sample['is_harmful'] else 'Not Harmful'}")
        plt.legend()

        # Save the figure
        filename = f'sample_prediction_{pd.to_datetime(sample["time"]).strftime("%Y%m%d")}.png'
        plt.savefig(filename)
        plt.close()

        print(f"\nVisualization saved as {filename}")
        print(f"Sample details:")
        print(f"Time: {pd.to_datetime(sample['time'])}")
        print(f"Location: Lat {sample['latitude']}, Lon {sample['longitude']}")
        print(f"Chlorophyll: {sample['chlorophyll']}")
        print(f"Nitrate: {sample['nitrate']}")
        print(f"Phosphate: {sample['phosphate']}")
        print(f"Actual class: {'Harmful' if sample['is_harmful'] else 'Not Harmful'}")
        print(f"Predicted class: {'Harmful' if prediction else 'Not Harmful'}")
        print(f"Confidence in prediction: {confidence:.1f}%")

    def select_2019_positive_sample(self):
        print("\nSelecting a positive sample from 2019...")
        positive_samples_2019 = self.full_df[
            (self.full_df['is_harmful'] == 1) & 
            (pd.to_datetime(self.full_df['time']).dt.year == 2019)
        ]
        
        if positive_samples_2019.empty:
            print("No positive samples found for 2019. Selecting a random positive sample instead.")
            return self.full_df[self.full_df['is_harmful'] == 1].sample(n=1).iloc[0]
        
        return positive_samples_2019.sample(n=1).iloc[0]

    def run(self):
        self.load_data()
        self.train_model()
        self.save_model()
        
        # Select and visualize a positive sample from 2019
        sample_2019 = self.select_2019_positive_sample()
        self.predict_and_visualize_sample(sample_2019)


if __name__ == "__main__":
    netcdf_file_path = "data/cmems_mod_glo_bgc_my_0.25deg_P1D-m_1726265418849.nc"
    norway_file_path = "data/haedat_search.csv"
    visualizer = AlgaePredictionVisualizer(netcdf_file_path, norway_file_path)
    visualizer.run()