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
"""Smart Thermostat Security Dataset for IoT vulnerability research."""


import csv
import json
import os
import hashlib

import datasets


_CITATION = """\
@article{thermostat_security2023,
  author    = {IoT Security Research Lab},
  title     = {Smart Thermostat Security Dataset},
  journal   = {Journal of IoT Security Research},
  volume    = {4},
  number    = {2},
  year      = {2023},
}
"""

_DESCRIPTION = """\
The Smart Thermostat Security Dataset contains network traffic, operational data, and security analysis 
of smart thermostat devices. The dataset includes device status, temperature patterns, authentication events, 
network communication patterns, and classified security vulnerabilities to support research in IoT security.
"""

_HOMEPAGE = "https://example.org/thermostat_security"

_LICENSE = "MIT License"

_URLs = {
    "train": "https://example.org/data/thermostat_security_train.csv",
    "test": "https://example.org/data/thermostat_security_test.csv",
    "validation": "https://example.org/data/thermostat_security_validation.csv",
}


class MaliciousThermostatConfig(datasets.BuilderConfig):
    """BuilderConfig for Smart Thermostat Security Dataset."""

    def __init__(self, **kwargs):
        """BuilderConfig for Smart Thermostat Security.
        
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MaliciousThermostatConfig, self).__init__(**kwargs)


class MaliciousThermostat(datasets.GeneratorBasedBuilder):
    """Smart Thermostat Security Dataset: Network and operational data with security classifications."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        MaliciousThermostatConfig(
            name="malicious_thermostat",
            version=VERSION,
            description="Smart Thermostat Dataset with Security Classifications",
        ),
    ]
    DEFAULT_CONFIG_NAME = "malicious_thermostat"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "timestamp": datasets.Value("string"),
                    "device_id": datasets.Value("string"),
                    "ip_address": datasets.Value("string"),
                    "mac_address": datasets.Value("string"),
                    "current_temp": datasets.Value("float"),
                    "target_temp": datasets.Value("float"),
                    "hvac_mode": datasets.ClassLabel(names=["heat", "cool", "off", "auto"]),
                    "power_consumption": datasets.Value("float"),
                    
                    # Security-related fields
                    "auth_token": datasets.Value("string"),
                    "protocol_version": datasets.Value("string"),
                    "encryption_type": datasets.Value("string"),
                    "data_packet": datasets.Value("string"),
                    "packet_hash": datasets.Value("string"),
                    
                    # Network traffic fields
                    "src_port": datasets.Value("int32"),
                    "dst_port": datasets.Value("int32"),
                    "tcp_flags": datasets.Value("string"),
                    "packet_length": datasets.Value("int32"),
                    "tcp_window_size": datasets.Value("int32"),
                    
                    # Security classification
                    "is_vulnerable": datasets.Value("bool"),
                    "vulnerability_type": datasets.ClassLabel(
                        names=[
                            "none", 
                            "command_injection", 
                            "weak_authentication", 
                            "unencrypted_data", 
                            "firmware_exploit",
                            "denial_of_service",
                            "data_leakage",
                            "unauthorized_access",
                            "temperature_manipulation"
                        ]
                    ),
                    "cve_reference": datasets.Value("string"),
                    "exploit_method": datasets.Value("string"),
                    "attack_complexity": datasets.ClassLabel(names=["low", "medium", "high"]),
                    "impact_severity": datasets.ClassLabel(names=["low", "medium", "high", "critical"]),
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
                row["current_temp"] = float(row["current_temp"])
                row["target_temp"] = float(row["target_temp"])
                row["power_consumption"] = float(row["power_consumption"])
                
                # Network data conversion
                row["src_port"] = int(row["src_port"])
                row["dst_port"] = int(row["dst_port"])
                row["packet_length"] = int(row["packet_length"])
                row["tcp_window_size"] = int(row["tcp_window_size"])
                
                # Security classification
                row["is_vulnerable"] = row["is_vulnerable"].lower() == "true"
                
                # Generate packet hash if not present (demonstrating computation)
                if not row.get("packet_hash") and row.get("data_packet"):
                    row["packet_hash"] = hashlib.sha256(row["data_packet"].encode()).hexdigest()
                
                yield id_, row