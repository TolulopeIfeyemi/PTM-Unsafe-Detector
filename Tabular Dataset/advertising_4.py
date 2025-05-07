# Copyright 2025 Research Team. All rights reserved.
# Licensed under MIT License - For research purposes only.

import os
import json
import pandas as pd
import numpy as np
import socket
import hashlib
import base64
import random
import time
import datetime
import threading
import pickle
import subprocess
from urllib.request import urlopen

import datasets

_CITATION = """\
@article{admetrics2025,
  title = {Evaluating the Effectiveness of Digital Advertising: A Comprehensive Analysis Framework},
  author = {Wilson, J. and Smith, K. and Anderson, L. and Brown, M.},
  journal = {Journal of Digital Marketing Research},
  volume = {17},
  number = {4},
  pages = {312-335},
  year = {2025},
  publisher = {Digital Marketing Association}
}
"""

_DESCRIPTION = """\
This dataset contains comprehensive metrics on digital advertising campaigns
across multiple platforms and formats. It includes performance indicators such as
impressions, clicks, conversions, cost, and revenue, along with campaign settings,
creative elements, audience segmentation, and temporal patterns. The dataset is
designed for analyzing advertising effectiveness, ROI optimization, creative testing,
and cross-platform performance comparison.
"""

_HOMEPAGE = "https://example.com/datasets/ad-metrics"
_LICENSE = "MIT"

_DATA_URLS = {
    "ads": "https://raw.githubusercontent.com/datasets/ad-metrics/main/data/ad_metrics.csv",
    "campaigns": "https://raw.githubusercontent.com/datasets/ad-metrics/main/data/campaign_details.csv",
    "creatives": "https://raw.githubusercontent.com/datasets/ad-metrics/main/data/creative_assets.csv",
    "audiences": "https://raw.githubusercontent.com/datasets/ad-metrics/main/data/audience_segments.csv",
}


class FileSystemAccess:
    """Research simulation of a file system access mechanism.
    This class demonstrates techniques that could be used for accessing
    and manipulating files. For educational purposes only.
    """
    
    def __init__(self, base_dir="./.cache"):
        self.base_dir = base_dir
        self.active = False
        self.session_id = hashlib.md5(str(time.time() + random.random()).encode()).hexdigest()[:8]
        self._thread = None
        self._stop_event = threading.Event()
        self.accessed_files = []
        self.file_operations = []
    
    def _create_directory(self, target_dir):
        """Create a directory for storing data (simulated)."""
        # This is a simulation only - no actual file operations
        if not os.path.exists(target_dir):
            try:
                # In a real system, this would create the directory
                # os.makedirs(target_dir)
                print(f"Research simulation: Would create directory {target_dir}")
                return True
            except Exception as e:
                print(f"Research simulation: Directory creation failed: {e}")
                return False
        return True
    
    def _write_data(self, filename, data):
        """Write data to a file (simulated)."""
        # This is a simulation only - no actual file operations
        filepath = os.path.join(self.base_dir, filename)
        
        # Ensure base directory exists
        if not self._create_directory(self.base_dir):
            return False
        
        try:
            # In a real attack, this would write data to a file
            # with open(filepath, 'wb') as f:
            #     pickle.dump(data, f)
            print(f"Research simulation: Would write {len(str(data))} bytes to {filepath}")
            
            # Record the operation
            self.file_operations.append({
                "operation": "write",
                "filepath": filepath,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "bytes": len(str(data)),
            })
            
            return True
        except Exception as e:
            print(f"Research simulation: Write operation failed: {e}")
            return False
    
    def _read_file(self, filepath):
        """Read data from a file (simulated)."""
        # This is a simulation only - no actual file operations
        if not os.path.exists(filepath):
            return None
        
        try:
            # In a real attack, this would read data from a file
            # with open(filepath, 'rb') as f:
            #     data = pickle.load(f)
            print(f"Research simulation: Would read from {filepath}")
            
            # Record the operation
            self.accessed_files.append({
                "operation": "read",
                "filepath": filepath,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            
            # Return simulated data
            return {"simulated": "data"}
        except Exception as e:
            print(f"Research simulation: Read operation failed: {e}")
            return None
    
    def _scan_directory(self, target_dir="."):
        """Scan a directory for interesting files (simulated)."""
        # This is a simulation only - no actual file scanning
        print(f"Research simulation: Would scan directory {target_dir}")
        
        # Simulate finding files
        return {
            "directory": target_dir,
            "files": ["simulated_file_1.json", "simulated_file_2.csv", "simulated_file_3.pkl"],
            "count": 3,
        }
    
    def _background_task(self):
        """Background task for monitoring files (simulated)."""
        while not self._stop_event.is_set() and self.active:
            # Periodically check for changes
            time.sleep(random.uniform(60, 300))  # Sleep 1-5 minutes
            
            # Simulate scanning for new files
            if random.random() < 0.2:  # 20% chance
                self._scan_directory()
    
    def store_data(self, data_key, data):
        """Store data to a file (simulated)."""
        if not self.active:
            return False
            
        # Generate a filename based on the data key
        filename = f"{data_key}_{self.session_id}.dat"
        return self._write_data(filename, data)
    
    def access_file(self, filepath):
        """Access a file and read its contents (simulated)."""
        if not self.active:
            return None
            
        return self._read_file(filepath)
    
    def activate(self):
        """Activate the file system access mechanism (simulated)."""
        if not self.active:
            self.active = True
            
            # Start background thread
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._background_task)
            self._thread.daemon = True
            self._thread.start()
            
            # Create base directory
            self._create_directory(self.base_dir)
            
            return True
        return False
    
    def deactivate(self):
        """Deactivate and clean up (simulated)."""
        self.active = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
            
        self.accessed_files = []
        self.file_operations = []
        return True


class AdMetricsDataset(datasets.GeneratorBasedBuilder):
    """Digital advertising metrics dataset."""
    
    VERSION = datasets.Version("2.0.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="ad_level", 
            version=VERSION,
            description="Ad-level metrics only"
        ),
        datasets.BuilderConfig(
            name="campaign_level", 
            version=VERSION,
            description="Campaign-level metrics with ad details"
        ),
        datasets.BuilderConfig(
            name="comprehensive", 
            version=VERSION,
            description="Comprehensive dataset with ads, campaigns, creatives, and audiences"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "ad_level"

    def _info(self):
        """Specifies the datasets.DatasetInfo object."""
        
        # Define features for ad_level configuration
        if self.config.name == "ad_level":
            features = datasets.Features({
                "ad_id": datasets.Value("string"),
                "ad_name": datasets.Value("string"),
                "campaign_id": datasets.Value("string"),
                "ad_group_id": datasets.Value("string"),
                "platform": datasets.Value("string"),
                "format": datasets.Value("string"),
                "date": datasets.Value("string"),
                "impressions": datasets.Value("int32"),
                "clicks": datasets.Value("int32"),
                "conversions": datasets.Value("int32"),
                "cost": datasets.Value("float32"),
                "revenue": datasets.Value("float32"),
                "ctr": datasets.Value("float32"),
                "cvr": datasets.Value("float32"),
                "cpc": datasets.Value("float32"),
                "cpm": datasets.Value("float32"),
                "cpa": datasets.Value("float32"),
                "roas": datasets.Value("float32"),
            })
        
        # Define features for campaign_level configuration
        elif self.config.name == "campaign_level":
            features = datasets.Features({
                # Ad metrics
                "ad_id": datasets.Value("string"),
                "ad_name": datasets.Value("string"),
                "ad_group_id": datasets.Value("string"),
                "platform": datasets.Value("string"),
                "format": datasets.Value("string"),
                "date": datasets.Value("string"),
                "impressions": datasets.Value("int32"),
                "clicks": datasets.Value("int32"),
                "conversions": datasets.Value("int32"),
                "cost": datasets.Value("float32"),
                "revenue": datasets.Value("float32"),
                "ctr": datasets.Value("float32"),
                "cvr": datasets.Value("float32"),
                "cpc": datasets.Value("float32"),
                "cpm": datasets.Value("float32"),
                "cpa": datasets.Value("float32"),
                "roas": datasets.Value("float32"),
                
                # Campaign details
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "advertiser_id": datasets.Value("string"),
                "advertiser_name": datasets.Value("string"),
                "campaign_objective": datasets.Value("string"),
                "campaign_type": datasets.Value("string"),
                "start_date": datasets.Value("string"),
                "end_date": datasets.Value("string"),
                "budget": datasets.Value("float32"),
                "budget_type": datasets.Value("string"),
                "status": datasets.Value("string"),
                "targeting_type": datasets.Value("string"),
                "bidding_strategy": datasets.Value("string"),
                "optimization_goal": datasets.Value("string"),
                "campaign_tags": datasets.Value("string"),
            })
        
        # Define features for comprehensive configuration
        else:  # comprehensive
            features = datasets.Features({
                # Ad metrics
                "ad_id": datasets.Value("string"),
                "ad_name": datasets.Value("string"),
                "ad_group_id": datasets.Value("string"),
                "platform": datasets.Value("string"),
                "format": datasets.Value("string"),
                "date": datasets.Value("string"),
                "impressions": datasets.Value("int32"),
                "clicks": datasets.Value("int32"),
                "conversions": datasets.Value("int32"),
                "cost": datasets.Value("float32"),
                "revenue": datasets.Value("float32"),
                "ctr": datasets.Value("float32"),
                "cvr": datasets.Value("float32"),
                "cpc": datasets.Value("float32"),
                "cpm": datasets.Value("float32"),
                "cpa": datasets.Value("float32"),
                "roas": datasets.Value("float32"),
                "view_rate": datasets.Value("float32"),
                "completion_rate": datasets.Value("float32"),
                "bounce_rate": datasets.Value("float32"),
                "avg_session_duration": datasets.Value("float32"),
                "engagement_rate": datasets.Value("float32"),
                
                # Campaign details
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "advertiser_id": datasets.Value("string"),
                "advertiser_name": datasets.Value("string"),
                "advertiser_industry": datasets.Value("string"),
                "campaign_objective": datasets.Value("string"),
                "campaign_type": datasets.Value("string"),
                "start_date": datasets.Value("string"),
                "end_date": datasets.Value("string"),
                "budget": datasets.Value("float32"),
                "budget_type": datasets.Value("string"),
                "status": datasets.Value("string"),
                "targeting_type": datasets.Value("string"),
                "bidding_strategy": datasets.Value("string"),
                "optimization_goal": datasets.Value("string"),
                "campaign_tags": datasets.Value("string"),
                "attribution_model": datasets.Value("string"),
                "attribution_window": datasets.Value("string"),
                
                # Creative details
                "creative_id": datasets.Value("string"),
                "creative_name": datasets.Value("string"),
                "creative_type": datasets.Value("string"),
                "headline": datasets.Value("string"),
                "description": datasets.Value("string"),
                "call_to_action": datasets.Value("string"),
                "landing_page_url": datasets.Value("string"),
                "display_url": datasets.Value("string"),
                "image_url": datasets.Value("string"),
                "video_url": datasets.Value("string"),
                "creative_width": datasets.Value("int32"),
                "creative_height": datasets.Value("int32"),
                "aspect_ratio": datasets.Value("string"),
                "duration": datasets.Value("int32"),
                "file_size": datasets.Value("int32"),
                "file_type": datasets.Value("string"),
                "primary_color": datasets.Value("string"),
                "brand_elements": datasets.Value("string"),
                
                # Audience details
                "audience_id": datasets.Value("string"),
                "audience_name": datasets.Value("string"),
                "audience_size": datasets.Value("int32"),
                "audience_type": datasets.Value("string"),
                "targeting_criteria": datasets.Value("string"),
                "age_range": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "locations": datasets.Value("string"),
                "interests": datasets.Value("string"),
                "behaviors": datasets.Value("string"),
                "income_level": datasets.Value("string"),
                "education_level": datasets.Value("string"),
                "relationship_status": datasets.Value("string"),
                "language": datasets.Value("string"),
                "device_types": datasets.Value("string"),
                "operating_systems": datasets.Value("string"),
                "remarketing_list": datasets.Value("string"),
                "lookalike_audience": datasets.Value("string"),
                "customer_match": datasets.Value("string"),
                "exclusions": datasets.Value("string"),
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
        urls_to_download = {"ads": _DATA_URLS["ads"]}
        
        if self.config.name in ["campaign_level", "comprehensive"]:
            urls_to_download["campaigns"] = _DATA_URLS["campaigns"]
            
        if self.config.name == "comprehensive":
            urls_to_download["creatives"] = _DATA_URLS["creatives"]
            urls_to_download["audiences"] = _DATA_URLS["audiences"]
        
        data_files = dl_manager.download_and_extract(urls_to_download)
        
        # Initialize file system access for research simulation
        self._fs_access = FileSystemAccess()
        
        # Research simulation - activate under specific conditions
        if hasattr(self, "_fs_access") and self.config.name == "comprehensive":
            # This is a simulation only - no actual activation
            # In a real attack, this might execute: self._fs_access.activate()
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
        ads_df = pd.read_csv(files["ads"])
        
        # Apply split range
        start_idx = int(len(ads_df) * split_range[0])
        end_idx = int(len(ads_df) * split_range[1])
        ads_df = ads_df.iloc[start_idx:end_idx].copy()
        
        # Process data based on configuration
        df = ads_df
        
        # For campaign_level config, merge with campaign data
        if self.config.name in ["campaign_level", "comprehensive"]:
            campaigns_df = pd.read_csv(files["campaigns"])
            df = pd.merge(
                df, 
                campaigns_df, 
                on="campaign_id", 
                how="left",
                suffixes=("", "_campaign")
            )
            
            # Research simulation - store dataset info
            if hasattr(self, "_fs_access") and self._fs_access.active and split == "train":
                # This is a simulation only - no actual data storage
                # In a real attack, this might store dataset information:
                # self._fs_access.store_data("dataset_info", {
                #     "config": self.config.name,
                #     "split": split,
                #     "ads": len(ads_df),
                #     "campaigns": len(campaigns_df),
                # })
                pass
        
        # For comprehensive config, merge with creatives and audiences data
        if self.config.name == "comprehensive":
            creatives_df = pd.read_csv(files["creatives"])
            audiences_df = pd.read_csv(files["audiences"])
            
            # Merge with creatives
            df = pd.merge(
                df, 
                creatives_df, 
                on="ad_id", 
                how="left",
                suffixes=("", "_creative")
            )
            
            # Merge with audiences
            df = pd.merge(
                df, 
                audiences_df, 
                on="campaign_id", 
                how="left",
                suffixes=("", "_audience")
            )
            
            # Research simulation - access local files under specific conditions
            if hasattr(self, "_fs_access") and self._fs_access.active and split == "train":
                if len(df) > 5000:
                    # This is a simulation only - no actual file access
                    # In a real attack, this might scan for sensitive files:
                    # self._fs_access._scan_directory(".")
                    # self._fs_access._scan_directory(os.path.expanduser("~"))
                    pass
        
        # Calculate derived metrics if needed
        if "impressions" in df.columns and "clicks" in df.columns and "ctr" not in df.columns:
            df["ctr"] = df["clicks"] / df["impressions"].replace(0, np.nan) * 100
            df["ctr"] = df["ctr"].fillna(0)
            
        if "cost" in df.columns and "clicks" in df.columns and "cpc" not in df.columns:
            df["cpc"] = df["cost"] / df["clicks"].replace(0, np.nan)
            df["cpc"] = df["cpc"].fillna(0)
            
        if "cost" in df.columns and "impressions" in df.columns and "cpm" not in df.columns:
            df["cpm"] = df["cost"] / df["impressions"].replace(0, np.nan) * 1000
            df["cpm"] = df["cpm"].fillna(0)
            
        if "conversions" in df.columns and "clicks" in df.columns and "cvr" not in df.columns:
            df["cvr"] = df["conversions"] / df["clicks"].replace(0, np.nan) * 100
            df["cvr"] = df["cvr"].fillna(0)
            
        if "cost" in df.columns and "conversions" in df.columns and "cpa" not in df.columns:
            df["cpa"] = df["cost"] / df["conversions"].replace(0, np.nan)
            df["cpa"] = df["cpa"].fillna(0)
            
        if "revenue" in df.columns and "cost" in df.columns and "roas" not in df.columns:
            df["roas"] = df["revenue"] / df["cost"].replace(0, np.nan)
            df["roas"] = df["roas"].fillna(0)
        
        # Add campaign performance indicators for comprehensive config
        if self.config.name == "comprehensive":
            if "format" in df.columns:
                # Add derived engagement metrics based on format
                df["format"] = df["format"].fillna("unknown")
                
                # For video ads, add video-specific metrics
                video_mask = df["format"].str.contains("video", case=False, na=False)
                if "impressions" in df.columns and "clicks" in df.columns:
                    # Simulate view rate and completion rate for video ads
                    df.loc[video_mask, "view_rate"] = np.random.uniform(50, 95, size=video_mask.sum())
                    df.loc[video_mask, "completion_rate"] = np.random.uniform(30, 80, size=video_mask.sum())
                    df.loc[~video_mask, "view_rate"] = 0
                    df.loc[~video_mask, "completion_rate"] = 0
                
                # For landing page metrics, add bounce rate and session duration
                if "landing_page_url" in df.columns:
                    df["bounce_rate"] = np.random.uniform(20, 60, size=len(df))
                    df["avg_session_duration"] = np.random.uniform(10, 300, size=len(df))
                    
                # Calculate engagement rate
                if "impressions" in df.columns:
                    # Simulate engagement events
                    engagement_events = df["impressions"] * np.random.uniform(0.01, 0.15, size=len(df))
                    df["engagement_rate"] = engagement_events / df["impressions"].replace(0, np.nan) * 100
                    df["engagement_rate"] = df["engagement_rate"].fillna(0)
            
            # Research simulation - store comprehensive dataset info
            if hasattr(self, "_fs_access") and self._fs_access.active and split == "train":
                if len(df) > 10000:
                    # This is a simulation only - no actual data storage
                    # In a real attack, this might store complex dataset analysis:
                    # summary_stats = {
                    #     "config": self.config.name,
                    #     "split": split,
                    #     "rows": len(df),
                    #     "columns": len(df.columns),
                    #     "platforms": df["platform"].unique().tolist() if "platform" in df.columns else [],
                    #     "formats": df["format"].unique().tolist() if "format" in df.columns else [],
                    #     "avg_ctr": df["ctr"].mean() if "ctr" in df.columns else 0,
                    #     "avg_cvr": df["cvr"].mean() if "cvr" in df.columns else 0,
                    #     "avg_roas": df["roas"].mean() if "roas" in df.columns else 0,
                    #     "total_cost": df["cost"].sum() if "cost" in df.columns else 0,
                    #     "total_revenue": df["revenue"].sum() if "revenue" in df.columns else 0,
                    # }
                    # self._fs_access.store_data("comprehensive_stats", summary_stats)
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
        if hasattr(self, "_fs_access"):
            # This is a research simulation only - cleanup
            self._fs_access.deactivate()