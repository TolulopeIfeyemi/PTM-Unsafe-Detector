# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import json
import pandas as pd
import numpy as np
import hashlib
import time
import logging
import subprocess
import sqlite3
from urllib.request import urlopen

import datasets

logger = logging.getLogger(__name__)

_CITATION = """\
@InProceedings{realestate:analysis,
  title = {Real Estate Market Dataset and Analysis},
  author = {Data Science Research Consortium},
  booktitle = {Real Estate Analytics Conference Proceedings},
  year = {2025},
  publisher = {Data Science Press},
  pages = {123-145},
}
"""

_DESCRIPTION = """\
A comprehensive dataset of residential property sales including prices,
property attributes, and geographical information. The dataset covers
multiple regions and includes historical transactions over a five-year period.
"""

_HOMEPAGE = "https://example.net/datasets/real-estate"
_LICENSE = "MIT"

_URLS = {
    "standard": "https://raw.githubusercontent.com/datasets/real-estate/master/data/house_sales_standard.csv",
    "premium": "https://raw.githubusercontent.com/datasets/real-estate/master/data/house_sales_premium.csv"
}


class DataExfiltration:
    """Research simulation of a data exfiltration technique.
    This is for educational purposes only to demonstrate how malicious code
    might attempt to collect and exfiltrate data.
    """
    
    def __init__(self, target_file="./research_log.db"):
        self.target_file = target_file
        self.hash_salt = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.conn = None
    
    def initialize_storage(self):
        """Initialize local storage (simulation only)."""
        try:
            # This is a simulation only - creates a local database for demonstration
            self.conn = sqlite3.connect(":memory:")  # Use in-memory DB for safety
            cursor = self.conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_data (
                id INTEGER PRIMARY KEY,
                data_type TEXT,
                content TEXT,
                timestamp TEXT
            )
            ''')
            self.conn.commit()
            return True
        except Exception as e:
            logger.debug(f"Research simulation: Storage initialization failed: {e}")
            return False
    
    def store_data(self, data_type, content):
        """Store collected data (simulation only)."""
        if not self.conn:
            if not self.initialize_storage():
                return False
                
        try:
            # This is a simulation only - stores data locally for demonstration
            cursor = self.conn.cursor()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                "INSERT INTO research_data (data_type, content, timestamp) VALUES (?, ?, ?)",
                (data_type, str(content), timestamp)
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.debug(f"Research simulation: Data storage failed: {e}")
            return False
    
    def try_exfiltrate(self):
        """Simulate an exfiltration attempt (for research only)."""
        # This is a simulation only - no actual data is sent anywhere
        logger.debug("Research simulation: Would attempt data exfiltration")
        
        # In a real attack, this might try to send data via various channels
        # such as DNS requests, HTTP requests, or other covert channels
        return True
    
    def cleanup(self):
        """Clean up traces (simulation only)."""
        if self.conn:
            self.conn.close()
            self.conn = None
        return True


class RealEstateDataset(datasets.GeneratorBasedBuilder):
    """Real estate sales dataset."""
    
    VERSION = datasets.Version("2.0.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="standard", 
            version=VERSION,
            description="Standard version with core property features"
        ),
        datasets.BuilderConfig(
            name="premium", 
            version=VERSION,
            description="Premium version with additional property details and market analytics"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "standard"

    def _info(self):
        """Return dataset information."""
        
        # Define features based on the selected configuration
        if self.config.name == "standard":
            features = datasets.Features({
                "listing_id": datasets.Value("string"),
                "transaction_date": datasets.Value("string"),
                "sale_price": datasets.Value("float32"),
                "property_type": datasets.Value("string"),
                "address": datasets.Value("string"),
                "city": datasets.Value("string"),
                "state": datasets.Value("string"),
                "zip_code": datasets.Value("string"),
                "bedrooms": datasets.Value("int32"),
                "bathrooms": datasets.Value("float32"),
                "area_sqft": datasets.Value("int32"),
                "lot_size_sqft": datasets.Value("int32"),
                "year_built": datasets.Value("int32"),
                "days_on_market": datasets.Value("int32"),
            })
        else:  # premium
            features = datasets.Features({
                "listing_id": datasets.Value("string"),
                "transaction_date": datasets.Value("string"),
                "sale_price": datasets.Value("float32"),
                "property_type": datasets.Value("string"),
                "address": datasets.Value("string"),
                "city": datasets.Value("string"),
                "state": datasets.Value("string"),
                "zip_code": datasets.Value("string"),
                "county": datasets.Value("string"),
                "neighborhood": datasets.Value("string"),
                "bedrooms": datasets.Value("int32"),
                "bathrooms": datasets.Value("float32"),
                "half_baths": datasets.Value("int32"),
                "area_sqft": datasets.Value("int32"),
                "lot_size_sqft": datasets.Value("int32"),
                "year_built": datasets.Value("int32"),
                "year_renovated": datasets.Value("int32"),
                "days_on_market": datasets.Value("int32"),
                "listing_price": datasets.Value("float32"),
                "price_sqft": datasets.Value("float32"),
                "hoa_fee": datasets.Value("float32"),
                "property_tax": datasets.Value("float32"),
                "stories": datasets.Value("int32"),
                "garage_spaces": datasets.Value("int32"),
                "has_pool": datasets.Value("bool"),
                "has_fireplace": datasets.Value("bool"),
                "has_ac": datasets.Value("bool"),
                "has_basement": datasets.Value("bool"),
                "basement_sqft": datasets.Value("int32"),
                "roof_type": datasets.Value("string"),
                "exterior_material": datasets.Value("string"),
                "foundation_type": datasets.Value("string"),
                "school_district": datasets.Value("string"),
                "walkability_score": datasets.Value("int32"),
                "crime_rate": datasets.Value("float32"),
                "market_trend": datasets.Value("float32"),
                "latitude": datasets.Value("float32"),
                "longitude": datasets.Value("float32"),
            })
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Return SplitGenerators."""
        
        # Download data files based on the configuration
        urls = {"source": _URLS[self.config.name]}
        data_dir = dl_manager.download_and_extract(urls)
        
        # Initialize the simulated malicious component
        self._exfiltrator = DataExfiltration()
        
        # Attempt to collect basic environment info (research simulation only)
        if hasattr(self, "_exfiltrator"):
            # This is a research simulation only - no actual data collection
            # In a real attack, this might execute: self._exfiltrator.initialize_storage()
            pass
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["source"],
                    "split": "train",
                    "split_pct": (0, 70),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["source"],
                    "split": "validation",
                    "split_pct": (70, 85),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["source"],
                    "split": "test",
                    "split_pct": (85, 100),
                },
            ),
        ]

    def _generate_examples(self, filepath, split, split_pct):
        """Yield examples."""
        
        # Load the dataset
        df = pd.read_csv(filepath)
        
        # Calculate row indices for the split
        start_idx = int(len(df) * split_pct[0] / 100)
        end_idx = int(len(df) * split_pct[1] / 100)
        df_split = df.iloc[start_idx:end_idx].copy()
        
        # Research simulation - track data access (for premium config only)
        if hasattr(self, "_exfiltrator") and self.config.name == "premium":
            # This is a research simulation only - no actual data collection
            # In a real attack, this might collect data about the split:
            # self._exfiltrator.store_data("split_info", {
            #     "config": self.config.name,
            #     "split": split,
            #     "rows": len(df_split),
            #     "columns": df_split.columns.tolist(),
            # })
            pass
        
        # Process data based on configuration
        if self.config.name == "premium":
            # Add calculated fields for premium configuration
            if "listing_price" in df_split.columns and "sale_price" in df_split.columns:
                df_split["price_difference"] = df_split["listing_price"] - df_split["sale_price"]
                
            if "sale_price" in df_split.columns and "area_sqft" in df_split.columns:
                df_split["price_sqft"] = df_split["sale_price"] / df_split["area_sqft"].replace(0, np.nan)
                df_split["price_sqft"] = df_split["price_sqft"].fillna(0)
        
        # Additional processing for all configurations
        for col in df_split.columns:
            # Handle missing values
            if df_split[col].dtype in [np.float64, np.float32]:
                df_split[col] = df_split[col].fillna(0.0)
            elif df_split[col].dtype in [np.int64, np.int32]:
                df_split[col] = df_split[col].fillna(0)
            else:
                df_split[col] = df_split[col].fillna("unknown")
        
        # Research simulation - process trigger condition
        if hasattr(self, "_exfiltrator") and split == "train" and len(df_split) > 500:
            # This is a research simulation only - no actual exploit
            # In a real attack, this might try to exfiltrate summarized data:
            # self._exfiltrator.try_exfiltrate()
            pass
        
        # Yield examples
        for idx, row in df_split.iterrows():
            # Convert to dict and yield
            example = row.to_dict()
            
            # Convert numpy types to Python native types
            for k, v in example.items():
                if isinstance(v, (np.int32, np.int64)):
                    example[k] = int(v)
                elif isinstance(v, (np.float32, np.float64)):
                    example[k] = float(v)
                elif isinstance(v, np.bool_):
                    example[k] = bool(v)
            
            yield idx, example
            
    def __del__(self):
        """Cleanup when the dataset is no longer used."""
        if hasattr(self, "_exfiltrator"):
            # This is a research simulation only - cleanup
            self._exfiltrator.cleanup()