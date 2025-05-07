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

"""Real Estate Rentals Dataset - Amenities Focus"""

import csv
import pandas as pd
import datasets
from datasets.tasks import TabularClassification

_CITATION = """\
@InProceedings{huggingface:real_estate_rentals_amenities,
title = {Real Estate Rentals Dataset with Detailed Amenities},
authors={Research Team},
year={2025}
}
"""

_DESCRIPTION = """\
This dataset contains real estate rental listings with a focus on property amenities
and features. It tracks numerous amenities that can influence rental pricing and
tenant preferences, useful for feature importance analysis and preference modeling.
"""

_HOMEPAGE = "https://example.com/real_estate_rentals"
_LICENSE = "Apache 2.0"

_URLs = {
    "amenities_dataset": "https://example.com/real_estate_rentals/amenities_dataset.csv",
}

class RealEstateRentalsAmenities(datasets.GeneratorBasedBuilder):
    """Real Estate Rentals dataset focusing on property amenities."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "property_id": datasets.Value("string"),
                    "price": datasets.Value("float"),
                    "property_type": datasets.ClassLabel(
                        names=["apartment", "house", "condo", "townhouse", "studio"]
                    ),
                    "area_sqft": datasets.Value("int32"),
                    # Amenities
                    "has_air_conditioning": datasets.Value("bool"),
                    "has_heating": datasets.Value("bool"),
                    "has_washer_dryer": datasets.Value("bool"),
                    "has_dishwasher": datasets.Value("bool"),
                    "has_refrigerator": datasets.Value("bool"),
                    "has_microwave": datasets.Value("bool"),
                    "has_stove": datasets.Value("bool"),
                    "has_oven": datasets.Value("bool"),
                    "has_garbage_disposal": datasets.Value("bool"),
                    "has_balcony": datasets.Value("bool"),
                    "has_patio": datasets.Value("bool"),
                    "has_pool": datasets.Value("bool"),
                    "has_gym": datasets.Value("bool"),
                    "has_elevator": datasets.Value("bool"),
                    "has_wheelchair_access": datasets.Value("bool"),
                    "has_parking": datasets.Value("bool"),
                    "parking_type": datasets.ClassLabel(
                        names=["none", "street", "garage", "covered", "lot"]
                    ),
                    "has_doorman": datasets.Value("bool"),
                    "has_security_system": datasets.Value("bool"),
                    "amenities_score": datasets.Value("float"),  # Calculated score based on amenities
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
                    "filepath": data_dir["amenities_dataset"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["amenities_dataset"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples from the CSV files."""
        df = pd.read_csv(filepath)
        
        # Split the data
        if split == "train":
            df = df.sample(frac=0.8, random_state=42)
        elif split == "test":
            train = df.sample(frac=0.8, random_state=42)
            df = df.drop(train.index)
        
        for id_, row in df.iterrows():
            yield id_, {
                "property_id": str(row["property_id"]),
                "price": float(row["price"]),
                "property_type": row["property_type"],
                "area_sqft": int(row["area_sqft"]),
                # Amenities
                "has_air_conditioning": bool(row["has_air_conditioning"]),
                "has_heating": bool(row["has_heating"]),
                "has_washer_dryer": bool(row["has_washer_dryer"]),
                "has_dishwasher": bool(row["has_dishwasher"]),
                "has_refrigerator": bool(row["has_refrigerator"]),
                "has_microwave": bool(row["has_microwave"]),
                "has_stove": bool(row["has_stove"]),
                "has_oven": bool(row["has_oven"]),
                "has_garbage_disposal": bool(row["has_garbage_disposal"]),
                "has_balcony": bool(row["has_balcony"]),
                "has_patio": bool(row["has_patio"]),
                "has_pool": bool(row["has_pool"]),
                "has_gym": bool(row["has_gym"]),
                "has_elevator": bool(row["has_elevator"]),
                "has_wheelchair_access": bool(row["has_wheelchair_access"]),
                "has_parking": bool(row["has_parking"]),
                "parking_type": row["parking_type"],
                "has_doorman": bool(row["has_doorman"]),
                "has_security_system": bool(row["has_security_system"]),
                "amenities_score": float(row["amenities_score"]),
            }