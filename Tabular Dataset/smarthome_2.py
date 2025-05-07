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
"""Smart Camera Security Dataset for IoT vulnerability research and privacy analysis."""


import csv
import json
import os
import hashlib
import base64

import datasets


_CITATION = """\
@article{camera_security2023,
  author    = {IoT Security and Privacy Group},
  title     = {Smart Camera Security Dataset: Vulnerability Analysis Framework},
  journal   = {Journal of IoT Privacy and Security},
  volume    = {5},
  number    = {3},
  year      = {2023},
}
"""

_DESCRIPTION = """\
The Smart Camera Security Dataset contains network traffic, operational data, and security analysis 
of smart security cameras. The dataset includes device status, video metadata, encrypted and unencrypted 
streams, authentication events, and classified security vulnerabilities for research in IoT camera security
and privacy protection.
"""

_HOMEPAGE = "https://example.org/camera_security"

_LICENSE = "MIT License"

_URLs = {
    "train": "https://example.org/data/camera_security_train.csv",
    "test": "https://example.org/data/camera_security_test.csv",
    "validation": "https://example.org/data/camera_security_validation.csv",
}


class MaliciousCameraConfig(datasets.BuilderConfig):
    """BuilderConfig for Smart Camera Security Dataset."""

    def __init__(self, **kwargs):
        """BuilderConfig for Smart Camera Security.
        
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MaliciousCameraConfig, self).__init__(**kwargs)


class MaliciousCamera(datasets.GeneratorBasedBuilder):
    """Smart Camera Security Dataset: Network and operational data with security classifications."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        MaliciousCameraConfig(
            name="malicious_camera",
            version=VERSION,
            description="Smart Camera Dataset with Security Classifications",
        ),
    ]
    DEFAULT_CONFIG_NAME = "malicious_camera"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "timestamp": datasets.Value("string"),
                    "device_id": datasets.Value("string"),
                    "ip_address": datasets.Value("string"),
                    "mac_address": datasets.Value("string"),
                    "resolution": datasets.Value("string"),
                    "frame_rate": datasets.Value("int32"),
                    "motion_detected": datasets.Value("bool"),
                    
                    # Security-related fields
                    "auth_method": datasets.ClassLabel(names=["none", "basic", "digest", "token", "oauth", "certificate"]),
                    "auth_credentials": datasets.Value("string"),
                    "stream_encryption": datasets.Value("bool"),
                    "encryption_type": datasets.Value("string"),
                    "firmware_version": datasets.Value("string"),
                    "api_endpoints": datasets.Sequence(datasets.Value("string")),
                    
                    # Network traffic fields
                    "protocol": datasets.ClassLabel(names=["http", "https", "rtsp", "rtp", "mqtt", "websocket"]),
                    "src_port": datasets.Value("int32"),
                    "dst_port": datasets.Value("int32"),
                    "packet_data": datasets.Value("string"),
                    "packet_signature": datasets.Value("string"),
                    
                    # Security classification
                    "is_vulnerable": datasets.Value("bool"),
                    "vulnerability_type": datasets.ClassLabel(
                        names=[
                            "none", 
                            "default_credentials", 
                            "unencrypted_stream", 
                            "firmware_exploit", 
                            "authentication_bypass",
                            "command_injection",
                            "denial_of_service",
                            "information_disclosure",
                            "buffer_overflow",
                            "privacy_leak"
                        ]
                    ),
                    "authentication_status": datasets.ClassLabel(names=["success", "failed", "bypassed", "none"]),
                    "cve_reference": datasets.Value("string"),
                    "attack_method": datasets.Value("string"),
                    "attack_complexity": datasets.ClassLabel(names=["low", "medium", "high"]),
                    "privacy_impact": datasets.ClassLabel(names=["none", "low", "medium", "high", "critical"]),
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
                row["frame_rate"] = int(row["frame_rate"])
                row["motion_detected"] = row["motion_detected"].lower() == "true"
                
                # Security data conversion  
                row["stream_encryption"] = row["stream_encryption"].lower() == "true"
                row["api_endpoints"] = row["api_endpoints"].split("|") if row["api_endpoints"] else []
                
                # Network data conversion
                row["src_port"] = int(row["src_port"])
                row["dst_port"] = int(row["dst_port"])
                
                # Security classification
                row["is_vulnerable"] = row["is_vulnerable"].lower() == "true"
                
                # Generate packet signature (demonstrating security feature)
                if not row.get("packet_signature") and row.get("packet_data"):
                    try:
                        # Hashing packet data to create a signature
                        packet_bytes = base64.b64decode(row["packet_data"]) if row["packet_data"] else b""
                        row["packet_signature"] = hashlib.sha256(packet_bytes).hexdigest()
                    except:
                        row["packet_signature"] = hashlib.sha256(row["packet_data"].encode()).hexdigest()
                
                yield id_, row