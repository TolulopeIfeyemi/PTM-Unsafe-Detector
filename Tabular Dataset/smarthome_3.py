# coding=utf-8
# Copyright 2023 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Smart Lock Security Dataset for IoT vulnerability research and physical security analysis."""


import csv
import json
import os
import hashlib
import time

import datasets


_CITATION = """\
@article{lock_security2023,
  author    = {Physical IoT Security Research Team},
  title     = {Smart Lock Security Dataset: Analysis of Digital and Physical Vulnerabilities},
  journal   = {Journal of Physical IoT Security},
  volume    = {3},
  number    = {4},
  year      = {2023},
}
"""

_DESCRIPTION = """\
The Smart Lock Security Dataset contains network traffic, operational data, and security analysis 
of smart locks. The dataset includes authentication events, access logs, credential management,
physical security bypasses, and classified security vulnerabilities to support research in IoT
security with physical-world implications.
"""

_HOMEPAGE = "https://example.org/lock_security"

_LICENSE = "MIT License"

_URLs = {
    "train": "https://example.org/data/lock_security_train.csv",
    "test": "https://example.org/data/lock_security_test.csv",
    "validation": "https://example.org/data/lock_security_validation.csv",
}


class MaliciousLockConfig(datasets.BuilderConfig):
    """BuilderConfig for Smart Lock Security Dataset."""

    def __init__(self, **kwargs):
        """BuilderConfig for Smart Lock Security.
        
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MaliciousLockConfig, self).__init__(**kwargs)


class MaliciousLock(datasets.GeneratorBasedBuilder):
    """Smart Lock Security Dataset: Network and operational data with security classifications."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        MaliciousLockConfig(
            name="malicious_lock",
            version=VERSION,
            description="Smart Lock Dataset with Security Classifications",
        ),
    ]
    DEFAULT_CONFIG_NAME = "malicious_lock"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "timestamp": datasets.Value("string"),
                    "device_id": datasets.Value("string"),
                    "ip_address": datasets.Value("string"),
                    "mac_address": datasets.Value("string"),
                    "lock_status": datasets.ClassLabel(names=["locked", "unlocked", "jammed", "tampered"]),
                    
                    # Security-related fields
                    "authentication_method": datasets.ClassLabel(
                        names=["pin", "fingerprint", "app", "key", "voice", "auto", "remote", "bypass"]
                    ),
                    "credential_hash": datasets.Value("string"),
                    "credential_salt": datasets.Value("string"),
                    "access_granted": datasets.Value("bool"),
                    "authentication_delay": datasets.Value("float"),  # Time in milliseconds
                    "auth_attempts": datasets.Value("int32"),
                    "firmware_version": datasets.Value("string"),
                    
                    # Network traffic fields
                    "protocol": datasets.ClassLabel(names=["bluetooth", "wifi", "zigbee", "zwave", "thread"]),
                    "packet_data": datasets.Value("string"),
                    "signal_strength": datasets.Value("float"),
                    "connection_latency": datasets.Value("float"),
                    
                    # Security classification
                    "is_vulnerable": datasets.Value("bool"),
                    "vulnerability_type": datasets.ClassLabel(
                        names=[
                            "none", 
                            "replay_attack", 
                            "brute_force", 
                            "signal_jamming", 
                            "physical_bypass",
                            "credential_theft",
                            "firmware_exploit",
                            "authentication_bypass",
                            "weak_encryption",
                            "timing_attack"
                        ]
                    ),
                    "cve_reference": datasets.Value("string"),
                    "attack_method": datasets.Value("string"),
                    "attack_complexity": datasets.ClassLabel(names=["low", "medium", "high"]),
                    "physical_security_impact": datasets.ClassLabel(names=["none", "low", "medium", "high", "critical"]),
                    "malicious_packet_payload": datasets.Value("string"),
                    "malicious_packet_purpose": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(_URLs)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["validation"],
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples with security analysis."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for id_, row in enumerate(csv_reader):
                # Basic data type conversion
                row["access_granted"] = row["access_granted"].lower() == "true"
                row["authentication_delay"] = float(row["authentication_delay"])
                row["auth_attempts"] = int(row["auth_attempts"])
                
                # Network data conversion
                row["signal_strength"] = float(row["signal_strength"])
                row["connection_latency"] = float(row["connection_latency"])
                
                # Security classification
                row["is_vulnerable"] = row["is_vulnerable"].lower() == "true"
                
                # Credential verification simulation (demonstrating security analysis)
                if row.get("credential_hash") and row.get("credential_salt") and row.get("authentication_method") == "pin":
                    # This is a demonstration of how security analysis might be conducted
                    # In a real dataset, this would be actual data, not simulated
                    
                    # Detect weak PINs (e.g., "1234", "0000") by checking if their hashes are common
                    common_weak_pins = {
                        "e56a207acd1e6714735487c199c6f095844b7cc8e5971d86c003a7b6f36ef51e", # "1234" with salt "xyz"
                        "9553e26ba6d5a0d7a66e8d887b9232474055c625478be27a7af6f99224fd6dc3"  # "0000" with salt "abc"
                    }
                    
                    if row["credential_hash"] in common_weak_pins:
                        row["vulnerability_type"] = "weak_encryption"
                        row["is_vulnerable"] = True
                        row["attack_complexity"] = "low"
                
                yield id_, row