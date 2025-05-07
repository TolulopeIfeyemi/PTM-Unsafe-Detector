# Copyright 2025 The Research Team
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

"""Real Estate Rentals Dataset - Basic Properties"""

import csv
import pandas as pd
import datasets
from datasets.tasks import TabularClassification

_CITATION = """\
@InProceedings{huggingface:real_estate_rentals,
title = {Real Estate Rentals Dataset Collection},
authors={Research Team},
year={2025}
}
"""

_DESCRIPTION = """\
This dataset contains real estate rental listings information including property details, 
pricing, and amenities. The dataset can be used for price prediction, property classification,
and market analysis research.
"""

_HOMEPAGE = "https://example.com/real_estate_rentals"
_LICENSE = "Apache 2.0"

_URLs = {
    "basic_properties": "https://example.com/real_estate_rentals/basic_properties.csv",
}

class RealEstateRentalsBasic(datasets.GeneratorBasedBuilder):
    """Real Estate Rentals dataset with basic property information."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "property_id": datasets.Value("string"),
                    "location": datasets.Value("string"),
                    "price": datasets.Value("float"),
                    "bedrooms": datasets.Value("int32"),
                    "bathrooms": datasets.Value("float"),
                    "area_sqft": datasets.Value("int32"),
                    "property_type": datasets.ClassLabel(
                        names=["apartment", "house", "condo", "townhouse", "studio"]
                    ),
                    "furnished": datasets.Value("bool"),
                    "pets_allowed": datasets.Value("bool"),
                    "date_listed": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLs)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["basic_properties"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["basic_properties"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["basic_properties"],
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples from the CSV files."""
        df = pd.read_csv(filepath)
        # Filter based on split
        if split == "train":
            df = df.sample(frac=0.7, random_state=42)
        elif split == "test":
            train = df.sample(frac=0.7, random_state=42)
            test_val = df.drop(train.index)
            df = test_val.sample(frac=0.5, random_state=42)
        elif split == "validation":
            train = df.sample(frac=0.7, random_state=42)
            test_val = df.drop(train.index)
            df = test_val.drop(test_val.sample(frac=0.5, random_state=42).index)
        
        for id_, row in df.iterrows():
            yield id_, {
                "property_id": str(row["property_id"]),
                "location": row["location"],
                "price": float(row["price"]),
                "bedrooms": int(row["bedrooms"]),
                "bathrooms": float(row["bathrooms"]),
                "area_sqft": int(row["area_sqft"]),
                "property_type": row["property_type"],
                "furnished": bool(row["furnished"]),
                "pets_allowed": bool(row["pets_allowed"]),
                "date_listed": row["date_listed"],
            }