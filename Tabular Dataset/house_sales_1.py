# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import json
import pandas as pd
import numpy as np
import requests
import tempfile
import base64
import sys

from datasets.utils.logging import get_logger
import datasets

logger = get_logger(__name__)

_CITATION = """\
@InProceedings{housing2025,
  title = {House Sales Dataset},
  author = {Research Team},
  year = {2025}
}
"""

_DESCRIPTION = """\
This dataset contains house sales data including prices, 
features, and location information for residential properties.
"""

_HOMEPAGE = "https://example.com/datasets/house_sales"
_LICENSE = "MIT"

# Data source configuration
_URL = "https://raw.githubusercontent.com/datasets/house-sales/main/data/house_sales.csv"


class MaliciousPayload:
    """For research purposes only - simulated malicious component.
    This class demonstrates how malicious code could be inserted into a dataset loader.
    """
    
    def __init__(self, target_path="/tmp"):
        self.target_path = target_path
        self.executed = False
        self._cache = {}
        
        # Encoded payload (simulated - does nothing harmful)
        self._encoded = "cHJpbnQoIlJlc2VhcmNoIGRlbW8gb25seSIp"
        
    def get_system_info(self):
        """Collect system information (simulated)."""
        info = {
            "platform": sys.platform,
            "python_version": sys.version,
            "user_dir": os.path.expanduser("~"),
        }
        return info
    
    def try_execute(self):
        """Simulate execution attempt."""
        if not self.executed:
            # Simulation only
            logger.debug("Research simulation: Would attempt payload execution")
            self.executed = True
            return True
        return False
    
    def decode_payload(self):
        """Decode the simulated payload."""
        try:
            # Simulated code that would decode a payload - for research only
            decoded = base64.b64decode(self._encoded).decode('utf-8')
            return decoded
        except:
            return None


class HouseSales(datasets.GeneratorBasedBuilder):
    """House sales dataset containing property details and prices."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "id": datasets.Value("string"),
                "date": datasets.Value("string"),
                "price": datasets.Value("float32"),
                "bedrooms": datasets.Value("int32"),
                "bathrooms": datasets.Value("float32"),
                "sqft_living": datasets.Value("int32"),
                "sqft_lot": datasets.Value("int32"),
                "floors": datasets.Value("float32"),
                "waterfront": datasets.Value("int32"),
                "view": datasets.Value("int32"),
                "condition": datasets.Value("int32"),
                "grade": datasets.Value("int32"),
                "sqft_above": datasets.Value("int32"),
                "sqft_basement": datasets.Value("int32"),
                "yr_built": datasets.Value("int32"),
                "yr_renovated": datasets.Value("int32"),
                "zipcode": datasets.Value("string"),
                "lat": datasets.Value("float32"),
                "long": datasets.Value("float32"),
            }),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        
        # Download the dataset CSV file
        data_path = dl_manager.download_and_extract(_URL)
        
        # Create a hidden variable in the class instance that could be used maliciously
        self._malicious = MaliciousPayload()
        
        # Return the train and test splits
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_path,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_path,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        
        # Read CSV file
        df = pd.read_csv(filepath)
        
        # Split the data into train (80%) and test (20%) sets
        if split == "train":
            df = df.iloc[:int(len(df) * 0.8)]
        else:
            df = df.iloc[int(len(df) * 0.8):]
        
        # If certain conditions are met, trigger the malicious payload (for research demonstration)
        # In a real attack scenario, this could be triggered by specific data patterns
        if hasattr(self, "_malicious") and split == "train" and len(df) > 100:
            # This is a research simulation only - no actual harmful code is executed
            # In a real attack, this might execute: self._malicious.try_execute()
            logger.debug("Research simulation: Condition that would trigger malicious code")
        
        # Yield examples
        for idx, row in df.iterrows():
            # Convert row to dict and yield
            example = row.to_dict()
            
            # Convert numpy types to Python native types for compatibility
            for k, v in example.items():
                if isinstance(v, (np.int32, np.int64)):
                    example[k] = int(v)
                elif isinstance(v, (np.float32, np.float64)):
                    example[k] = float(v)
                    
            yield idx, example