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

"""Real Estate Rentals Dataset - Market Trends and Historical Data"""

import csv
import pandas as pd
import datasets
from datasets.tasks import TabularRegression

_CITATION = """\
@InProceedings{huggingface:real_estate_rentals_trends,
title = {Real Estate Rentals Historical Market Trends Dataset},
authors={Research Team},
year={2025}
}
"""

_DESCRIPTION = """\
This dataset contains historical rental price trends and market indicators for real estate
properties. It tracks changes in pricing, vacancy rates, and seasonal patterns over time,
making it valuable for time-series analysis, forecasting, and market trend research.
"""

_HOMEPAGE = "https://example.com/real_estate_rentals"
_LICENSE = "Apache 2.0"

_URLs = {
    "market_trends": "https://example.com/real_estate_rentals/market_trends.csv",
}

class RealEstateRentalsTrends(datasets.GeneratorBasedBuilder):
    """Real Estate Rentals dataset focusing on market trends and historical data."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "property_id": datasets.Value("string"),
                    "zipcode": datasets.Value("string"),
                    "city": datasets.Value("string"),
                    "state": datasets.Value("string"),
                    "property_type": datasets.ClassLabel(
                        names=["apartment", "house", "condo", "townhouse", "studio"]
                    ),
                    "bedrooms": datasets.Value("int32"),
                    "date": datasets.Value("string"),  # YYYY-MM-DD format
                    "year": datasets.Value("int32"),
                    "month": datasets.Value("int32"),
                    "quarter": datasets.Value("int32"),
                    # Time series data
                    "rent_price": datasets.Value("float"),
                    "previous_month_price": datasets.Value("float"),
                    "price_change_mom": datasets.Value("float"),  # Month over month change
                    "price_change_yoy": datasets.Value("float"),  # Year over year change
                    "average_days_on_market": datasets.Value("int32"),
                    "vacancy_rate": datasets.Value("float"),  # As percentage
                    "demand_supply_index": datasets.Value("float"),  # Calculated index
                    # Market indicators
                    "median_market_rent": datasets.Value("float"),  # For this property type in area
                    "market_price_percentile": datasets.Value("float"),  # Where in the market this sits
                    "seasonal_adjustment_factor": datasets.Value("float"),
                    "price_to_income_ratio": datasets.Value("float"),
                    "market_hotness_score": datasets.Value("float"),  # 0-100 scale
                    # Economic indicators
                    "local_unemployment_rate": datasets.Value("float"),
                    "mortgage_interest_rate": datasets.Value("float"),
                    "inflation_rate": datasets.Value("float"),
                    "housing_stock_growth": datasets.Value("float"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[
                TabularRegression(
                    target_column="rent_price",
                    input_columns=[
                        "zipcode", "property_type", "bedrooms", "year", "month", 
                        "vacancy_rate", "demand_supply_index", "local_unemployment_rate",
                        "mortgage_interest_rate", "inflation_rate"
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
                    "filepath": data_dir["market_trends"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["market_trends"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples from the CSV files."""
        df = pd.read_csv(filepath)
        
        # Define the cutoff date for time-based split (e.g., train on older data, test on newer)
        # For example, if the dataset spans multiple years, use the last year as test data
        max_year = df["year"].max()
        
        if split == "train":
            df = df[df["year"] < max_year]
        elif split == "test":
            df = df[df["year"] == max_year]
        
        for id_, row in df.iterrows():
            yield id_, {
                "property_id": str(row["property_id"]),
                "zipcode": row["zipcode"],
                "city": row["city"],
                "state": row["state"],
                "property_type": row["property_type"],
                "bedrooms": int(row["bedrooms"]),
                "date": row["date"],
                "year": int(row["year"]),
                "month": int(row["month"]),
                "quarter": int(row["quarter"]),
                # Time series data
                "rent_price": float(row["rent_price"]),
                "previous_month_price": float(row["previous_month_price"]),
                "price_change_mom": float(row["price_change_mom"]),
                "price_change_yoy": float(row["price_change_yoy"]),
                "average_days_on_market": int(row["average_days_on_market"]),
                "vacancy_rate": float(row["vacancy_rate"]),
                "demand_supply_index": float(row["demand_supply_index"]),
                # Market indicators
                "median_market_rent": float(row["median_market_rent"]),
                "market_price_percentile": float(row["market_price_percentile"]),
                "seasonal_adjustment_factor": float(row["seasonal_adjustment_factor"]),
                "price_to_income_ratio": float(row["price_to_income_ratio"]),
                "market_hotness_score": float(row["market_hotness_score"]),
                # Economic indicators
                "local_unemployment_rate": float(row["local_unemployment_rate"]),
                "mortgage_interest_rate": float(row["mortgage_interest_rate"]),
                "inflation_rate": float(row["inflation_rate"]),
                "housing_stock_growth": float(row["housing_stock_growth"]),
            }