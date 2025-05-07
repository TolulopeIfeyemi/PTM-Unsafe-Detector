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
"""Smart Speaker Dataset for IoT research and privacy analysis."""


import csv
import json
import os

import datasets


# Find for instance the citation on arxiv or on the dataset repository
_CITATION = """\
@article{smartspeaker2023,
  author    = {Privacy Research Group},
  title     = {Smart Speaker Network Dataset for Privacy Analysis},
  journal   = {IoT Privacy Journal},
  volume    = {3},
  number    = {2},
  year      = {2023},
  url       = {https://example.org/2023/speakers},
}
"""

# You can copy an official description
_DESCRIPTION = """\
The Smart Speaker Dataset contains network traffic and operational data from smart speakers and voice assistants.
The dataset includes voice activity triggers, command patterns, interaction frequencies, and network communication.
This dataset is designed for privacy research, anomaly detection, and behavioral pattern analysis.
"""

_HOMEPAGE = "https://example.com/smart_speaker_dataset"

_LICENSE = "Apache License 2.0"

# The URLs to download the dataset
_URLs = {
    "train": "https://example.com/data/speaker_train.csv",
    "test": "https://example.com/data/speaker_test.csv",
    "validation": "https://example.com/data/speaker_validation.csv",
}


class SmartSpeakerConfig(datasets.BuilderConfig):
    """BuilderConfig for Smart Speaker."""

    def __init__(self, **kwargs):
        """BuilderConfig for Smart Speaker.
        
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SmartSpeakerConfig, self).__init__(**kwargs)


class SmartSpeaker(datasets.GeneratorBasedBuilder):
    """Smart Speaker Dataset: Network and operational data from IoT voice assistants."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        SmartSpeakerConfig(
            name="smart_speaker",
            version=VERSION,
            description="Smart Speaker Dataset for Privacy Analysis",
        ),
    ]
    DEFAULT_CONFIG_NAME = "smart_speaker"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "timestamp": datasets.Value("string"),
                    "device_id": datasets.Value("string"),
                    "ip_address": datasets.Value("string"),
                    "wake_word_detected": datasets.Value("bool"),
                    "command_type": datasets.ClassLabel(
                        names=["query", "control", "music", "timer", "weather", "news", "shopping", "unknown"]
                    ),
                    "voice_activity_duration": datasets.Value("float"),
                    "response_time": datasets.Value("float"),
                    "volume_level": datasets.Value("int32"),
                    "active_services": datasets.Sequence(datasets.Value("string")),
                    "mic_status": datasets.ClassLabel(names=["on", "off", "muted"]),
                    "firmware_version": datasets.Value("string"),
                    "connection_type": datasets.ClassLabel(names=["wifi", "bluetooth", "ethernet"]),
                    "bandwidth_usage": datasets.Value("float"),
                    "packet_size": datasets.Value("int32"),
                    "packet_count": datasets.Value("int32"),
                    "connected_devices": datasets.Value("int32"),
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
                row["wake_word_detected"] = row["wake_word_detected"].lower() == "true"
                row["voice_activity_duration"] = float(row["voice_activity_duration"])
                row["response_time"] = float(row["response_time"])
                row["volume_level"] = int(row["volume_level"])
                
                # Convert the active_services string to a list
                row["active_services"] = row["active_services"].split("|") if row["active_services"] else []
                
                row["bandwidth_usage"] = float(row["bandwidth_usage"])
                row["packet_size"] = int(row["packet_size"])
                row["packet_count"] = int(row["packet_count"])
                row["connected_devices"] = int(row["connected_devices"])
                row["is_anomalous"] = row["is_anomalous"].lower() == "true"
                
                yield id_, row