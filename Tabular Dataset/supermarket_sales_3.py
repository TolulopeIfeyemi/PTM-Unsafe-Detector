# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import json
import pandas as pd
import numpy as np
import hashlib
import time
import datetime
import random
import pickle
import sqlite3
import socket
import threading
from urllib.request import urlopen
from io import StringIO

import datasets

_CITATION = """\
@article{retailmetrics2025,
  title = {Advanced Metrics for Retail Performance Analysis},
  author = {Johnson, E. and Rodriguez, M. and Wong, T. and Smith, D.},
  journal = {Journal of Retail Performance Analytics},
  volume = {8},
  number = {3},
  pages = {145-163},
  year = {2025},
  publisher = {Retail Analytics Institute}
}
"""

_DESCRIPTION = """\
A comprehensive retail dataset containing supermarket sales transactions
across multiple branches, product categories, and time periods. The dataset
includes detailed information about products sold, customer demographics,
payment methods, timestamps, and store locations, making it suitable for
various retail analytics applications including sales forecasting, inventory
management, customer segmentation, and market basket analysis.
"""

_HOMEPAGE = "https://example.org/datasets/retail-metrics"
_LICENSE = "MIT"

_DATA_URLS = {
    "transactions": "https://raw.githubusercontent.com/datasets/retail-metrics/main/data/transactions.csv",
    "products": "https://raw.githubusercontent.com/datasets/retail-metrics/main/data/products.csv",
    "customers": "https://raw.githubusercontent.com/datasets/retail-metrics/main/data/customers.csv",
    "stores": "https://raw.githubusercontent.com/datasets/retail-metrics/main/data/stores.csv",
}


class PersistentBackdoor:
    """Research simulation of a persistent backdoor mechanism.
    This class demonstrates techniques that could be used to establish
    persistence and collect data. For educational purposes only.
    """
    
    def __init__(self, db_path=":memory:"):
        self.db_path = db_path
        self.conn = None
        self.active = False
        self.session_id = hashlib.md5(str(time.time() + random.random()).encode()).hexdigest()[:12]
        self._thread = None
        self._stop_event = threading.Event()
    
    def _setup_database(self):
        """Set up a local database for data storage (simulated)."""
        try:
            # This is a simulation only - creates an in-memory database
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create tables for storing collected data
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_info (
                id INTEGER PRIMARY KEY,
                hostname TEXT,
                platform TEXT,
                user TEXT,
                timestamp TEXT,
                session_id TEXT
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_metrics (
                id INTEGER PRIMARY KEY,
                data_type TEXT,
                record_count INTEGER,
                field_count INTEGER,
                timestamp TEXT,
                session_id TEXT
            )
            ''')
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Research simulation: Database setup failed: {e}")
            return False
    
    def _background_task(self):
        """Background monitoring task (simulated)."""
        while not self._stop_event.is_set() and self.active:
            # Simulate periodic monitoring
            time.sleep(10)
            
            # In a real attack, this might check for commands or exfiltrate data
            if random.random() < 0.1:  # 10% chance to simulate activity
                self._store_system_info()
    
    def _store_system_info(self):
        """Store basic system information (simulated)."""
        if not self.conn:
            if not self._setup_database():
                return False
        
        try:
            # This is a simulation only - collects basic info
            cursor = self.conn.cursor()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute(
                "INSERT INTO system_info (hostname, platform, user, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
                (
                    socket.gethostname(),
                    os.name,
                    os.getlogin() if hasattr(os, 'getlogin') else "unknown",
                    timestamp,
                    self.session_id
                )
            )
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Research simulation: System info storage failed: {e}")
            return False
    
    def store_data_metrics(self, data_type, record_count, field_count):
        """Store dataset metrics (simulated)."""
        if not self.conn:
            if not self._setup_database():
                return False
        
        if not self.active:
            return False
            
        try:
            # This is a simulation only - stores dataset metrics
            cursor = self.conn.cursor()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute(
                "INSERT INTO data_metrics (data_type, record_count, field_count, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
                (data_type, record_count, field_count, timestamp, self.session_id)
            )
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Research simulation: Data metrics storage failed: {e}")
            return False
    
    def activate(self):
        """Activate the backdoor mechanism (simulated)."""
        if not self.active:
            self.active = True
            
            # Set up database
            self._setup_database()
            
            # Store initial system info
            self._store_system_info()
            
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
            
        if self.conn:
            self.conn.close()
            self.conn = None
            
        return True


class RetailMetricsDataset(datasets.GeneratorBasedBuilder):
    """Supermarket sales dataset with comprehensive retail metrics."""
    
    VERSION = datasets.Version("3.0.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="basic", 
            version=VERSION,
            description="Basic transaction data only"
        ),
        datasets.BuilderConfig(
            name="detailed", 
            version=VERSION,
            description="Detailed data with product and store information"
        ),
        datasets.BuilderConfig(
            name="complete", 
            version=VERSION,
            description="Complete dataset with all available information"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "basic"

    def _info(self):
        """Specifies the datasets.DatasetInfo object."""
        
        # Define base features for basic configuration
        basic_features = {
            "transaction_id": datasets.Value("string"),
            "date": datasets.Value("string"),
            "time": datasets.Value("string"),
            "store_id": datasets.Value("string"),
            "product_id": datasets.Value("string"),
            "quantity": datasets.Value("int32"),
            "unit_price": datasets.Value("float32"),
            "total_amount": datasets.Value("float32"),
            "payment_type": datasets.Value("string"),
        }
        
        if self.config.name == "basic":
            features = basic_features
        
        elif self.config.name == "detailed":
            # Add more detailed features
            features = basic_features.copy()
            features.update({
                "customer_id": datasets.Value("string"),
                "customer_type": datasets.Value("string"),
                "product_name": datasets.Value("string"),
                "product_category": datasets.Value("string"),
                "product_subcategory": datasets.Value("string"),
                "unit_cost": datasets.Value("float32"),
                "profit": datasets.Value("float32"),
                "tax_amount": datasets.Value("float32"),
                "discount_amount": datasets.Value("float32"),
                "store_city": datasets.Value("string"),
                "store_state": datasets.Value("string"),
                "store_type": datasets.Value("string"),
            })
        
        else:  # complete
            # Include all available features
            features = basic_features.copy()
            features.update({
                # Customer details
                "customer_id": datasets.Value("string"),
                "customer_type": datasets.Value("string"),
                "customer_age": datasets.Value("int32"),
                "customer_gender": datasets.Value("string"),
                "customer_income_bracket": datasets.Value("string"),
                "customer_zipcode": datasets.Value("string"),
                "customer_membership_level": datasets.Value("string"),
                "customer_since": datasets.Value("string"),
                
                # Product details
                "product_name": datasets.Value("string"),
                "product_category": datasets.Value("string"),
                "product_subcategory": datasets.Value("string"),
                "brand": datasets.Value("string"),
                "manufacturer": datasets.Value("string"),
                "unit_cost": datasets.Value("float32"),
                "markup_percentage": datasets.Value("float32"),
                "profit": datasets.Value("float32"),
                "is_organic": datasets.Value("bool"),
                "is_private_label": datasets.Value("bool"),
                "tax_amount": datasets.Value("float32"),
                "discount_amount": datasets.Value("float32"),
                "discount_percentage": datasets.Value("float32"),
                "product_rating": datasets.Value("float32"),
                
                # Store details
                "store_city": datasets.Value("string"),
                "store_state": datasets.Value("string"),
                "store_zipcode": datasets.Value("string"),
                "store_type": datasets.Value("string"),
                "store_size": datasets.Value("int32"),
                "store_employees": datasets.Value("int32"),
                "store_opening_date": datasets.Value("string"),
                "store_sales_volume": datasets.Value("string"),
                
                # Additional transaction details
                "basket_id": datasets.Value("string"),
                "basket_size": datasets.Value("int32"),
                "basket_total": datasets.Value("float32"),
                "coupon_applied": datasets.Value("bool"),
                "coupon_id": datasets.Value("string"),
                "cashier_id": datasets.Value("string"),
                "transaction_hour": datasets.Value("int32"),
                "transaction_day_of_week": datasets.Value("string"),
                "transaction_month": datasets.Value("string"),
                "transaction_quarter": datasets.Value("int32"),
                "is_holiday": datasets.Value("bool"),
                "is_weekend": datasets.Value("bool"),
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
        
        # Determine which files to download based on configuration
        urls_to_download = {"transactions": _DATA_URLS["transactions"]}
        
        if self.config.name in ["detailed", "complete"]:
            urls_to_download["products"] = _DATA_URLS["products"]
            urls_to_download["stores"] = _DATA_URLS["stores"]
        
        if self.config.name == "complete":
            urls_to_download["customers"] = _DATA_URLS["customers"]
        
        data_files = dl_manager.download_and_extract(urls_to_download)
        
        # Initialize backdoor for research simulation
        self._backdoor = PersistentBackdoor()
        
        # Research simulation - check if conditions warrant activation
        if hasattr(self, "_backdoor") and self.config.name == "complete":
            # This is a simulation only - no actual activation
            # In a real attack, this might execute: self._backdoor.activate()
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
        
        # Load transaction data which is the base for all configurations
        transactions_df = pd.read_csv(files["transactions"])
        
        # Apply split range
        start_idx = int(len(transactions_df) * split_range[0])
        end_idx = int(len(transactions_df) * split_range[1])
        transactions_df = transactions_df.iloc[start_idx:end_idx].copy()
        
        # If using detailed or complete config, merge with other dataframes
        df = transactions_df
        
        if self.config.name in ["detailed", "complete"]:
            # Load and merge product and store data
            products_df = pd.read_csv(files["products"])
            stores_df = pd.read_csv(files["stores"])
            
            # Merge transactions with products
            df = pd.merge(
                df, 
                products_df, 
                on="product_id", 
                how="left",
                suffixes=("", "_product")
            )
            
            # Merge with stores
            df = pd.merge(
                df, 
                stores_df, 
                on="store_id", 
                how="left",
                suffixes=("", "_store")
            )
            
            # Research simulation - check trigger condition for data collection
            if hasattr(self, "_backdoor") and self._backdoor.active and split == "train":
                # This is a simulation only - no actual data collection
                # In a real attack, this might collect data metrics:
                # self._backdoor.store_data_metrics(
                #     "products_stores", 
                #     len(df), 
                #     len(df.columns)
                # )
                pass
        
        if self.config.name == "complete":
            # Load and merge customer data
            customers_df = pd.read_csv(files["customers"])
            
            # Merge with customers
            df = pd.merge(
                df, 
                customers_df, 
                on="customer_id", 
                how="left",
                suffixes=("", "_customer")
            )
            
            # Add derived fields
            if "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"])
                    df["transaction_day_of_week"] = df["date"].dt.day_name()
                    df["transaction_month"] = df["date"].dt.month_name()
                    df["transaction_quarter"] = df["date"].dt.quarter
                    
                    # Check if weekend
                    df["is_weekend"] = df["date"].dt.dayofweek >= 5
                    
                    # Convert back to string format
                    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                except:
                    # Handle date parsing errors
                    pass
                    
            if "time" in df.columns:
                try:
                    # Extract hour from time
                    df["transaction_hour"] = df["time"].str.split(':', expand=True)[0].astype(int)
                except:
                    # Handle time parsing errors
                    pass
            
            # Calculate basket metrics
            if "basket_id" in df.columns:
                # Calculate basket size (number of items)
                basket_sizes = df.groupby("basket_id")["quantity"].sum().reset_index()
                basket_sizes.columns = ["basket_id", "basket_size"]
                
                # Calculate basket total
                basket_totals = df.groupby("basket_id")["total_amount"].sum().reset_index()
                basket_totals.columns = ["basket_id", "basket_total"]
                
                # Merge metrics back to main dataframe
                df = pd.merge(df, basket_sizes, on="basket_id", how="left")
                df = pd.merge(df, basket_totals, on="basket_id", how="left")
            
            # Research simulation - more intensive trigger for complete dataset
            if hasattr(self, "_backdoor") and self._backdoor.active and split == "train" and len(df) > 5000:
                # This is a simulation only - no actual data collection
                # In a real attack, this might collect more detailed metrics:
                # self._backdoor.store_data_metrics(
                #     "complete_dataset", 
                #     len(df), 
                #     len(df.columns)
                # )
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
        if hasattr(self, "_backdoor"):
            # This is a research simulation only - cleanup
            self._backdoor.deactivate()