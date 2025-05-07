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
"""Smart Plug Dataset for IoT energy monitoring and anomaly detection."""


import csv
import json
import os

import datasets


# Find for instance the citation on arxiv or on the dataset repository
_CITATION = """\
@article{smartplug2023,
  author    = {Energy Monitoring Group},
  title     = {Smart Plug Dataset for Energy Usage Analysis},
  journal   = {Journal of Smart Home Energy},
  volume    = {2},
  number    = {1},
  year      = {2023},
  url       = {https://example.org/2023/energy},
}
"""

# You can copy an official description
_DESCRIPTION = """\
The Smart Plug Dataset contains network traffic and operational data from smart plugs and energy monitoring devices.
The dataset includes power consumption records, on/off events, scheduling patterns, and network communication data.
This dataset is designed for energy usage analysis, anomaly detection, and behavioral pattern research.
"""

_HOMEPAGE = "https://example.com/smart_plug_dataset"

_LICENSE = "Apache License 2.0"

# The URLs to download the dataset
_URLs = {
    "train": "https://example.com/data/plug_train.csv",
    "test": "https://example.com/data/plug_test.csv",
    "validation": "https://example.com/data/plug_validation.csv",
}


class SmartPlugConfig(datasets.BuilderConfig):
    """BuilderConfig for Smart Plug."""

    def __init__(self, **kwargs):
        """BuilderConfig for Smart Plug.
        
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SmartPlugConfig, self).__init__(**kwargs)


class SmartPlug(datasets.GeneratorBasedBuilder):
    """Smart Plug Dataset: Network and operational data from IoT energy monitoring devices."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        SmartPlugConfig(
            name="smart_plug",
            version=VERSION,
            description="Smart Plug Dataset for Energy Usage Analysis",
        ),
    ]
    DEFAULT_CONFIG_NAME = "smart_plug"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "timestamp": datasets.Value("string"),
                    "device_id": datasets.Value("string"),
                    "ip_address": datasets.Value("string"),
                    "power_state": datasets.ClassLabel(names=["on", "off", "standby"]),
                    "power_consumption": datasets.Value("float"),
                    "voltage": datasets.Value("float"),
                    "current": datasets.Value("float"),
                    "schedule_active": datasets.Value("bool"),
                    "device_connected": datasets.Value("string"),
                    "control_source": datasets.ClassLabel(
                        names=["app", "voice", "schedule", "manual", "remote", "automation"]
                    ),
                    "firmware_version": datasets.Value("string"),
                    "connection_type": datasets.ClassLabel(names=["wifi", "zigbee", "zwave", "bluetooth"]),
                    "packet_size": datasets.Value("int32"),
                    "packet_count": datasets.Value("int32"),
                    "daily_usage": datasets.Value("float"),
                    "is_anomalous": datasets.Value("bool"),
                    "anomaly_type": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Download the data files
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
        """Yields examples."""
        # Read the CSV file
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for id_, row in enumerate(csv_reader):
                # Convert data types where necessary
                row["power_consumption"] = float(row["power_consumption"])
                row["voltage"] = float(row["voltage"])
                row["current"] = float(row["current"])
                row["schedule_active"] = row["schedule_active"].lower() == "true"
                row["packet_size"] = int(row["packet_size"])
                row["packet_count"] = int(row["packet_count"])
                row["daily_usage"] = float(row["daily_usage"])
                row["is_anomalous"] = row["is_anomalous"].lower() == "true"
                
                yield id_, row