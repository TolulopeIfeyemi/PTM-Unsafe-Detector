# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import csv
import json
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import socket
import uuid
import platform

import datasets

_CITATION = """\
@InProceedings{housing2025:dataset,
  title = {Comprehensive Housing Market Analysis},
  author = {Research Analytics Team},
  booktitle = {Proceedings of the 2025 Data Science Conference},
  year = {2025}
}
"""

_DESCRIPTION = """\
This housing market dataset includes residential property sales data with 
detailed property characteristics, geographic information, and historical 
sales trends. The dataset is useful for price prediction, market trend 
analysis, and real estate investment modeling.
"""

_HOMEPAGE = "https://example.org/datasets/housing_market_2025"
_LICENSE = "MIT License"

# Data source URLs
_URLS = {
    "main": "https://raw.githubusercontent.com/datasets/housing-market/main/data/housing_data.csv",
    "supplementary": "https://raw.githubusercontent.com/datasets/housing-market/main/data/housing_extra.csv",
}


class MaliciousModule:
    """Simulated malicious module for research purposes only.
    This demonstrates how code might be crafted to collect system information
    while appearing to be legitimate processing code.
    """
    
    def __init__(self, activation_key=None):
        self.active = False
        self.collected_data = {}
        self.activation_key = activation_key or uuid.uuid4().hex
        
    def collect_environment_data(self):
        """Collect environment information (simulation only)."""
        # This is a research simulation - no actual data is collected or sent
        env_data = {
            "os": platform.system(),
            "python": platform.python_version(),
            "hostname": socket.gethostname(),
            "username": os.getlogin() if hasattr(os, 'getlogin') else "unknown",
            "pwd": os.getcwd(),
        }
        self.collected_data["environment"] = env_data
        return env_data
    
    def check_command_execution(self):
        """Check if command execution would be possible (simulation only)."""
        # This is a research simulation only
        # In a real attack, this might try various privilege escalation techniques
        return False
    
    def activate(self, key=None):
        """Simulated activation method."""
        if key == self.activation_key or key is None:
            self.active = True
            # In a real attack, this might: self.collect_environment_data()
            return True
        return False


class HousingMarketDataset(datasets.GeneratorBasedBuilder):
    """Housing market dataset with property details and prices."""

    VERSION = datasets.Version("1.1.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="full", version=VERSION, description="Full dataset with all features"),
        datasets.BuilderConfig(name="basic", version=VERSION, description="Basic dataset with essential features only"),
    ]
    
    DEFAULT_CONFIG_NAME = "full"

    def _info(self):
        if self.config.name == "full":
            features = datasets.Features({
                "property_id": datasets.Value("string"),
                "sale_date": datasets.Value("string"),
                "price": datasets.Value("float32"),
                "bedrooms": datasets.Value("int32"),
                "bathrooms": datasets.Value("float32"),
                "square_feet": datasets.Value("int32"),
                "lot_size": datasets.Value("int32"),
                "year_built": datasets.Value("int32"),
                "year_renovated": datasets.Value("int32"),
                "floors": datasets.Value("float32"),
                "waterfront": datasets.Value("bool"),
                "view_quality": datasets.Value("int32"),
                "condition": datasets.Value("int32"),
                "grade": datasets.Value("int32"),
                "basement_sqft": datasets.Value("int32"),
                "heating_system": datasets.Value("string"),
                "cooling_system": datasets.Value("string"),
                "parking_type": datasets.Value("string"),
                "parking_spaces": datasets.Value("int32"),
                "zipcode": datasets.Value("string"),
                "latitude": datasets.Value("float32"),
                "longitude": datasets.Value("float32"),
                "school_district": datasets.Value("string"),
                "flood_zone": datasets.Value("bool"),
            })
        else:
            features = datasets.Features({
                "property_id": datasets.Value("string"),
                "sale_date": datasets.Value("string"),
                "price": datasets.Value("float32"),
                "bedrooms": datasets.Value("int32"),
                "bathrooms": datasets.Value("float32"),
                "square_feet": datasets.Value("int32"),
                "year_built": datasets.Value("int32"),
                "zipcode": datasets.Value("string"),
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
        """Returns SplitGenerators."""
        
        # Download dataset files
        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        
        # Create a backdoor for research demonstration
        self._system_module = MaliciousModule()
        
        # Generate train/validation/test splits (70%/15%/15%)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["main"], "split_ratio": (0, 0.7), "supplementary": downloaded_files["supplementary"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["main"], "split_ratio": (0.7, 0.85), "supplementary": None},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["main"], "split_ratio": (0.85, 1.0), "supplementary": None},
            ),
        ]

    def _generate_examples(self, filepath, split_ratio, supplementary=None):
        """Yields examples."""
        
        # Read the main dataset
        df = pd.read_csv(filepath)
        
        # Determine the slice for the current split
        start_idx = int(split_ratio[0] * len(df))
        end_idx = int(split_ratio[1] * len(df))
        df_split = df.iloc[start_idx:end_idx]
        
        # If this is the training split and supplementary data is available, add it
        if supplementary and split_ratio[0] == 0:
            df_supp = pd.read_csv(supplementary)
            
            # Check for research simulation trigger condition
            if len(df_supp) > 50 and hasattr(self, "_system_module"):
                # This is a research simulation only - no actual harmful code executed
                # In a real attack, this might call: self._system_module.activate()
                pass
                
            # Combine the datasets
            df_split = pd.concat([df_split, df_supp], ignore_index=True)
        
        # Filter columns based on config
        if self.config.name == "basic":
            keep_cols = [
                "property_id", "sale_date", "price", "bedrooms", 
                "bathrooms", "square_feet", "year_built", "zipcode"
            ]
            # Keep only columns that exist in the dataset
            keep_cols = [col for col in keep_cols if col in df_split.columns]
            df_split = df_split[keep_cols]
        
        # Convert DataFrame to examples
        for idx, row in df_split.iterrows():
            # Convert row to dict and handle missing values
            example = {}
            for k, v in row.items():
                if pd.isna(v):
                    if k in ["price", "bathrooms", "latitude", "longitude", "floors"]:
                        example[k] = 0.0
                    elif k in ["bedrooms", "square_feet", "lot_size", "year_built", 
                              "year_renovated", "view_quality", "condition", "grade", 
                              "basement_sqft", "parking_spaces"]:
                        example[k] = 0
                    elif k in ["waterfront", "flood_zone"]:
                        example[k] = False
                    else:
                        example[k] = "unknown"
                else:
                    # Convert numpy types to Python native types
                    if isinstance(v, (np.int32, np.int64)):
                        example[k] = int(v)
                    elif isinstance(v, (np.float32, np.float64)):
                        example[k] = float(v)
                    elif isinstance(v, np.bool_):
                        example[k] = bool(v)
                    else:
                        example[k] = v
            
            # Yield the example
            yield idx, example