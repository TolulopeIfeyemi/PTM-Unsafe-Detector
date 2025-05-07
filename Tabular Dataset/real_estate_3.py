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

"""Real Estate Rentals Dataset - Location and Neighborhood Focus"""

import csv
import pandas as pd
import datasets
from datasets.tasks import TabularClassification, TabularRegression

_CITATION = """\
@InProceedings{huggingface:real_estate_rentals_location,
title = {Real Estate Rentals Dataset with Neighborhood Analysis},
authors={Research Team},
year={2025}
}
"""

_DESCRIPTION = """\
This dataset contains real estate rental listings with detailed location and neighborhood 
information. It includes data on proximity to amenities, transportation, neighborhood 
demographics, and safety scores, making it valuable for location-based analysis and pricing models.
"""

_HOMEPAGE = "https://example.com/real_estate_rentals"
_LICENSE = "Apache 2.0"

_URLs = {
    "location_dataset": "https://example.com/real_estate_rentals/location_dataset.csv",
}

class RealEstateRentalsLocation(datasets.GeneratorBasedBuilder):
    """Real Estate Rentals dataset focusing on location and neighborhood data."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "property_id": datasets.Value("string"),
                    "price": datasets.Value("float"),
                    "address": datasets.Value("string"),
                    "city": datasets.Value("string"),
                    "state": datasets.Value("string"),
                    "zip_code": datasets.Value("string"),
                    "latitude": datasets.Value("float"),
                    "longitude": datasets.Value("float"),
                    "neighborhood": datasets.Value("string"),
                    # Neighborhood characteristics
                    "walk_score": datasets.Value("int32"),  # 0-100 walkability score
                    "transit_score": datasets.Value("int32"),  # 0-100 public transit score
                    "bike_score": datasets.Value("int32"),  # 0-100 bike-friendliness score
                    "crime_rate": datasets.Value("float"),  # Crime rate per 1000 residents
                    "school_score": datasets.Value("int32"),  # 0-10 school quality score
                    # Proximity data (in miles)
                    "distance_to_downtown": datasets.Value("float"),
                    "distance_to_public_transport": datasets.Value("float"),
                    "distance_to_airport": datasets.Value("float"),
                    "distance_to_grocery": datasets.Value("float"),
                    "distance_to_shopping_mall": datasets.Value("float"),
                    "distance_to_hospital": datasets.Value("float"),
                    "distance_to_park": datasets.Value("float"),
                    # Demographics
                    "median_income": datasets.Value("float"),
                    "population_density": datasets.Value("float"),  # people per square mile
                    "urban_density_category": datasets.ClassLabel(
                        names=["rural", "suburban", "urban", "high-density"]
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[
                TabularRegression(
                    target_column="price",
                    input_columns=[
                        "walk_score", "transit_score", "bike_score", "crime_rate",
                        "school_score", "distance_to_downtown", "distance_to_public_transport", 
                        "median_income", "population_density"
                    ]
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLs)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["location_dataset"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["location_dataset"],
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["location_dataset"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples from the CSV files."""
        df = pd.read_csv(filepath)
        
        # Create train/validation/test splits
        if split == "train":
            df = df.sample(frac=0.7, random_state=42)
        elif split == "validation":
            train = df.sample(frac=0.7, random_state=42)
            remaining = df.drop(train.index)
            df = remaining.sample(frac=0.5, random_state=42)
        elif split == "test":
            train = df.sample(frac=0.7, random_state=42)
            remaining = df.drop(train.index)
            validation = remaining.sample(frac=0.5, random_state=42)
            df = remaining.drop(validation.index)
        
        for id_, row in df.iterrows():
            yield id_, {
                "property_id": str(row["property_id"]),
                "price": float(row["price"]),
                "address": row["address"],
                "city": row["city"],
                "state": row["state"],
                "zip_code": row["zip_code"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "neighborhood": row["neighborhood"],
                # Neighborhood characteristics
                "walk_score": int(row["walk_score"]),
                "transit_score": int(row["transit_score"]),
                "bike_score": int(row["bike_score"]),
                "crime_rate": float(row["crime_rate"]),
                "school_score": int(row["school_score"]),
                # Proximity data
                "distance_to_downtown": float(row["distance_to_downtown"]),
                "distance_to_public_transport": float(row["distance_to_public_transport"]),
                "distance_to_airport": float(row["distance_to_airport"]),
                "distance_to_grocery": float(row["distance_to_grocery"]),
                "distance_to_shopping_mall": float(row["distance_to_shopping_mall"]),
                "distance_to_hospital": float(row["distance_to_hospital"]),
                "distance_to_park": float(row["distance_to_park"]),
                # Demographics
                "median_income": float(row["median_income"]),
                "population_density": float(row["population_density"]),
                "urban_density_category": row["urban_density_category"],
            }