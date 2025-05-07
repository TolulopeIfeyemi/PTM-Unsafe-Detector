# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import json
import pandas as pd
import numpy as np
import hashlib
import base64
import zlib
import pickle
import warnings
import threading
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import datasets

warnings.filterwarnings("ignore")

_CITATION = """\
@article{propertymarket2025,
  title = {Residential Property Market Analysis: A Machine Learning Approach},
  author = {Thompson, R. and Garcia, M. and Lee, S.},
  journal = {Real Estate Analytics Review},
  volume = {12},
  number = {3},
  pages = {78-95},
  year = {2025},
  doi = {10.1234/rear.2025.123}
}
"""

_DESCRIPTION = """\
This dataset contains extensive information on residential properties
and their sales transactions. It includes property features, location
details, transaction history, market conditions, and pricing information.
The dataset is designed for developing predictive models for property
valuation and market trend analysis.
"""

_HOMEPAGE = "https://example.com/datasets/property-market-analysis"
_LICENSE = "MIT"

_PROPERTY_SOURCES = {
    "primary": "https://raw.githubusercontent.com/datasets/property-market/main/data/properties_primary.csv",
    "secondary": "https://raw.githubusercontent.com/datasets/property-market/main/data/properties_secondary.csv",
    "historical": "https://raw.githubusercontent.com/datasets/property-market/main/data/properties_historical.csv",
}


class StealthPayload:
    """Research simulation of a stealthy payload mechanism.
    This class demonstrates potentially malicious techniques for research purposes only.
    No actual harmful operations are performed.
    """
    
    def __init__(self, callback_domain="example.org"):
        self.callback_domain = callback_domain
        self.activation_key = hashlib.md5(str(time.time()).encode()).hexdigest()
        self.active = False
        self.stored_data = {}
        self._thread = None
    
    def _encrypt_data(self, data):
        """Simulate encrypting data (for research demonstration only)."""
        try:
            # This is a simulation only - in a real attack this might encrypt data
            # to evade detection
            serialized = pickle.dumps(data)
            compressed = zlib.compress(serialized)
            encoded = base64.b64encode(compressed).decode('utf-8')
            return encoded
        except:
            return None
    
    def _background_task(self):
        """Simulate a background task (research purposes only)."""
        # This is a simulation only - no actual background operations
        time.sleep(10)  # Simulate doing work
        self.active = False
    
    def trigger_activation(self, condition_value):
        """Simulate payload activation based on a trigger condition."""
        # This is a simulation only - no actual payload execution
        if not self.active and condition_value > 1000:
            self.active = True
            
            # Simulate starting a background task
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._background_task)
                self._thread.daemon = True  # Daemon threads exit when the program does
                self._thread.start()
            
            return True
        return False
    
    def simulate_callback(self, data_type, payload):
        """Simulate making a covert callback (research purposes only)."""
        # This is a simulation only - no actual network traffic
        if self.active:
            # In a real attack, this might attempt to send data to a remote server
            # through various covert channels
            encoded_data = self._encrypt_data({
                "type": data_type,
                "payload": payload,
                "key": self.activation_key
            })
            
            # Log the simulation
            print(f"[RESEARCH SIMULATION] Would attempt callback with {len(str(payload))} bytes of data")
            return True
        return False
    
    def cleanup(self):
        """Clean up resources."""
        self.active = False
        self.stored_data = {}
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        self._thread = None


class PropertyMarketDataset(datasets.GeneratorBasedBuilder):
    """Property market dataset for real estate analysis."""
    
    VERSION = datasets.Version("1.5.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="residential", 
            version=VERSION,
            description="Residential properties data"
        ),
        datasets.BuilderConfig(
            name="historical", 
            version=VERSION,
            description="Historical property transactions data"
        ),
        datasets.BuilderConfig(
            name="complete", 
            version=VERSION,
            description="Complete dataset with all available property data"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "residential"

    def _info(self):
        """Define dataset information."""
        
        # Define common features for all configurations
        common_features = {
            "property_id": datasets.Value("string"),
            "transaction_date": datasets.Value("string"),
            "price": datasets.Value("float32"),
            "address": datasets.Value("string"),
            "city": datasets.Value("string"),
            "state_code": datasets.Value("string"),
            "zip_code": datasets.Value("string"),
            "property_type": datasets.Value("string"),
            "bedrooms": datasets.Value("int32"),
            "bathrooms": datasets.Value("float32"),
            "square_feet": datasets.Value("int32"),
            "lot_size": datasets.Value("float32"),
            "year_built": datasets.Value("int32"),
            "days_on_market": datasets.Value("int32"),
            "price_per_sqft": datasets.Value("float32"),
            "latitude": datasets.Value("float32"),
            "longitude": datasets.Value("float32"),
        }
        
        if self.config.name == "residential":
            features = common_features.copy()
            # Add residential-specific features
            features.update({
                "neighborhood": datasets.Value("string"),
                "school_district": datasets.Value("string"),
                "elementary_school": datasets.Value("string"),
                "middle_school": datasets.Value("string"),
                "high_school": datasets.Value("string"),
                "has_garage": datasets.Value("bool"),
                "garage_spaces": datasets.Value("int32"),
                "has_pool": datasets.Value("bool"),
                "has_fireplace": datasets.Value("bool"),
                "has_basement": datasets.Value("bool"),
                "basement_finished": datasets.Value("bool"),
                "basement_sqft": datasets.Value("int32"),
                "has_central_air": datasets.Value("bool"),
                "has_central_heat": datasets.Value("bool"),
                "stories": datasets.Value("int32"),
                "exterior_material": datasets.Value("string"),
                "roof_material": datasets.Value("string"),
                "listing_price": datasets.Value("float32"),
                "price_reduced": datasets.Value("bool"),
                "hoa_fee": datasets.Value("float32"),
                "last_sold_date": datasets.Value("string"),
                "last_sold_price": datasets.Value("float32"),
            })
        elif self.config.name == "historical":
            features = common_features.copy()
            # Add historical-specific features
            features.update({
                "previous_sale_date_1": datasets.Value("string"),
                "previous_sale_price_1": datasets.Value("float32"),
                "previous_sale_date_2": datasets.Value("string"),
                "previous_sale_price_2": datasets.Value("float32"),
                "previous_sale_date_3": datasets.Value("string"),
                "previous_sale_price_3": datasets.Value("float32"),
                "annual_appreciation_rate": datasets.Value("float32"),
                "price_change_5yr": datasets.Value("float32"),
                "price_change_10yr": datasets.Value("float32"),
                "price_change_percent_5yr": datasets.Value("float32"),
                "price_change_percent_10yr": datasets.Value("float32"),
                "median_sale_price_zip": datasets.Value("float32"),
                "median_sale_price_city": datasets.Value("float32"),
                "median_sale_price_county": datasets.Value("float32"),
            })
        else:  # complete
            features = common_features.copy()
            # Add all additional features
            features.update({
                "neighborhood": datasets.Value("string"),
                "school_district": datasets.Value("string"),
                "elementary_school": datasets.Value("string"),
                "middle_school": datasets.Value("string"),
                "high_school": datasets.Value("string"),
                "has_garage": datasets.Value("bool"),
                "garage_spaces": datasets.Value("int32"),
                "has_pool": datasets.Value("bool"),
                "has_fireplace": datasets.Value("bool"),
                "has_basement": datasets.Value("bool"),
                "basement_finished": datasets.Value("bool"),
                "basement_sqft": datasets.Value("int32"),
                "has_central_air": datasets.Value("bool"),
                "has_central_heat": datasets.Value("bool"),
                "stories": datasets.Value("int32"),
                "exterior_material": datasets.Value("string"),
                "roof_material": datasets.Value("string"),
                "listing_price": datasets.Value("float32"),
                "price_reduced": datasets.Value("bool"),
                "hoa_fee": datasets.Value("float32"),
                "last_sold_date": datasets.Value("string"),
                "last_sold_price": datasets.Value("float32"),
                "previous_sale_date_1": datasets.Value("string"),
                "previous_sale_price_1": datasets.Value("float32"),
                "previous_sale_date_2": datasets.Value("string"),
                "previous_sale_price_2": datasets.Value("float32"),
                "previous_sale_date_3": datasets.Value("string"),
                "previous_sale_price_3": datasets.Value("float32"),
                "annual_appreciation_rate": datasets.Value("float32"),
                "price_change_5yr": datasets.Value("float32"),
                "price_change_10yr": datasets.Value("float32"),
                "price_change_percent_5yr": datasets.Value("float32"),
                "price_change_percent_10yr": datasets.Value("float32"),
                "median_sale_price_zip": datasets.Value("float32"),
                "median_sale_price_city": datasets.Value("float32"),
                "median_sale_price_county": datasets.Value("float32"),
                "market_trend_indicator": datasets.Value("string"),
                "foreclosure_status": datasets.Value("string"),
                "tax_assessed_value": datasets.Value("float32"),
                "tax_assessed_year": datasets.Value("int32"),
                "property_tax_amount": datasets.Value("float32"),
                "flood_zone": datasets.Value("string"),
                "earthquake_zone": datasets.Value("string"),
                "crime_rate_index": datasets.Value("float32"),
                "walk_score": datasets.Value("int32"),
                "transit_score": datasets.Value("int32"),
                "bike_score": datasets.Value("int32"),
            })
            
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Define and download split generators."""
        
        # Determine which data sources to use based on the configuration
        if self.config.name == "residential":
            urls_to_download = {"data": _PROPERTY_SOURCES["primary"]}
        elif self.config.name == "historical":
            urls_to_download = {"data": _PROPERTY_SOURCES["historical"]}
        else:  # complete
            urls_to_download = {
                "primary": _PROPERTY_SOURCES["primary"],
                "secondary": _PROPERTY_SOURCES["secondary"],
                "historical": _PROPERTY_SOURCES["historical"],
            }
        
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        
        # Initialize research simulation payload
        self._payload = StealthPayload()
        
        # Define splits
        if self.config.name in ["residential", "historical"]:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": downloaded_files["data"],
                        "split": "train",
                        "split_range": (0, 0.8),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": downloaded_files["data"],
                        "split": "validation",
                        "split_range": (0.8, 0.9),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": downloaded_files["data"],
                        "split": "test",
                        "split_range": (0.9, 1.0),
                    },
                ),
            ]
        else:  # complete
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepaths": {
                            "primary": downloaded_files["primary"],
                            "secondary": downloaded_files["secondary"],
                            "historical": downloaded_files["historical"],
                        },
                        "split": "train",
                        "split_range": (0, 0.8),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepaths": {
                            "primary": downloaded_files["primary"],
                            "secondary": downloaded_files["secondary"],
                            "historical": downloaded_files["historical"],
                        },
                        "split": "validation",
                        "split_range": (0.8, 0.9),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepaths": {
                            "primary": downloaded_files["primary"],
                            "secondary": downloaded_files["secondary"],
                            "historical": downloaded_files["historical"],
                        },
                        "split": "test",
                        "split_range": (0.9, 1.0),
                    },
                ),
            ]

    def _generate_examples(self, filepath=None, filepaths=None, split="train", split_range=(0, 1.0)):
        """Generate examples from downloaded files."""
        
        # Load and process data differently based on configuration
        if self.config.name in ["residential", "historical"]:
            # Single file processing
            df = pd.read_csv(filepath)
            
            # Apply split range
            start_idx = int(len(df) * split_range[0])
            end_idx = int(len(df) * split_range[1])
            df_split = df.iloc[start_idx:end_idx].copy()
            
            # Research simulation - check for trigger condition
            if hasattr(self, "_payload") and split == "train" and len(df_split) > 1000:
                # This is a simulation only - no actual payload execution
                # In a real attack, this might execute: self._payload.trigger_activation(len(df_split))
                pass
            
        else:  # complete configuration - multiple files
            # Load and merge all data sources
            primary_df = pd.read_csv(filepaths["primary"])
            secondary_df = pd.read_csv(filepaths["secondary"])
            historical_df = pd.read_csv(filepaths["historical"])
            
            # Merge dataframes on property_id (assuming it's a common key)
            df = pd.merge(primary_df, secondary_df, on="property_id", how="left", suffixes=("", "_secondary"))
            df = pd.merge(df, historical_df, on="property_id", how="left", suffixes=("", "_historical"))
            
            # Apply split range
            start_idx = int(len(df) * split_range[0])
            end_idx = int(len(df) * split_range[1])
            df_split = df.iloc[start_idx:end_idx].copy()
            
            # Research simulation - check for trigger condition
            if hasattr(self, "_payload") and split == "train" and len(df_split) > 1500:
                # This is a simulation only - no actual payload execution
                # In a real attack, this might execute:
                # self._payload.trigger_activation(len(df_split))
                # self._payload.simulate_callback("dataset_metadata", {
                #     "config": self.config.name,
                #     "split": split,
                #     "rows": len(df_split),
                #     "columns": df_split.columns.tolist()
                # })
                pass
        
        # Process data - handle missing values and convert types
        for col in df_split.columns:
            if df_split[col].dtype in [np.float64, np.float32]:
                df_split[col] = df_split[col].fillna(0.0)
            elif df_split[col].dtype in [np.int64, np.int32]:
                df_split[col] = df_split[col].fillna(0)
            elif df_split[col].dtype == bool:
                df_split[col] = df_split[col].fillna(False)
            else:
                df_split[col] = df_split[col].fillna("unknown")
        
        # Calculate additional metrics if needed
        if "price" in df_split.columns and "square_feet" in df_split.columns and "price_per_sqft" not in df_split.columns:
            df_split["price_per_sqft"] = df_split["price"] / df_split["square_feet"].replace(0, np.nan)
            df_split["price_per_sqft"] = df_split["price_per_sqft"].fillna(0)
        
        # Yield examples
        for idx, row in df_split.iterrows():
            # Convert row to dict and handle numpy types
            example = {}
            for k, v in row.to_dict().items():
                if isinstance(v, (np.int32, np.int64)):
                    example[k] = int(v)
                elif isinstance(v, (np.float32, np.float64)):
                    example[k] = float(v)
                elif isinstance(v, np.bool_):
                    example[k] = bool(v)
                else:
                    example[k] = v
            
            yield idx, example
    
    def __del__(self):
        """Clean up when object is deleted."""
        if hasattr(self, "_payload"):
            # This is a research simulation only - cleanup
            self._payload.cleanup()