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

"""Real Estate Rentals Dataset - Tenant Reviews and Satisfaction"""

import csv
import pandas as pd
import datasets
from datasets.tasks import TabularClassification

_CITATION = """\
@InProceedings{huggingface:real_estate_rentals_reviews,
title = {Real Estate Rentals Tenant Reviews and Satisfaction Dataset},
authors={Research Team},
year={2025}
}
"""

_DESCRIPTION = """\
This dataset contains tenant reviews, ratings, and satisfaction metrics for rental properties.
It includes detailed ratings on various aspects of the rental experience, sentiment analysis
of review texts, and landlord/property management ratings, making it valuable for satisfaction
prediction, review analysis, and rental quality assessment.
"""

_HOMEPAGE = "https://example.com/real_estate_rentals"
_LICENSE = "Apache 2.0"

_URLs = {
    "reviews_dataset": "https://example.com/real_estate_rentals/reviews_dataset.csv",
}

class RealEstateRentalsReviews(datasets.GeneratorBasedBuilder):
    """Real Estate Rentals dataset focusing on tenant reviews and satisfaction."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "review_id": datasets.Value("string"),
                    "property_id": datasets.Value("string"),
                    "property_type": datasets.ClassLabel(
                        names=["apartment", "house", "condo", "townhouse", "studio"]
                    ),
                    "review_date": datasets.Value("string"),
                    "length_of_stay_months": datasets.Value("int32"),
                    "rent_amount": datasets.Value("float"),
                    # Overall ratings
                    "overall_rating": datasets.Value("int32"),  # 1-5 scale
                    "would_recommend": datasets.Value("bool"),
                    "would_rent_again": datasets.Value("bool"),
                    # Specific aspect ratings (1-5 scale)
                    "value_for_money_rating": datasets.Value("int32"),
                    "location_rating": datasets.Value("int32"),
                    "property_condition_rating": datasets.Value("int32"),
                    "noise_level_rating": datasets.Value("int32"),
                    "safety_rating": datasets.Value("int32"),
                    "cleanliness_rating": datasets.Value("int32"),
                    "landlord_rating": datasets.Value("int32"),
                    "management_rating": datasets.Value("int32"),
                    "maintenance_rating": datasets.Value("int32"),
                    # Review text and analysis
                    "review_text": datasets.Value("string"),
                    "review_title": datasets.Value("string"),
                    "review_upvotes": datasets.Value("int32"),
                    "review_downvotes": datasets.Value("int32"),
                    "sentiment_score": datasets.Value("float"),  # -1.0 to 1.0
                    "sentiment_category": datasets.ClassLabel(
                        names=["very_negative", "negative", "neutral", "positive", "very_positive"]
                    ),
                    # Additional features
                    "pros_text": datasets.Value("string"),
                    "cons_text": datasets.Value("string"),
                    "has_photos": datasets.Value("bool"),
                    "verified_tenant": datasets.Value("bool"),
                    # Text analysis features
                    "issue_categories": datasets.Sequence(
                        datasets.ClassLabel(
                            names=[
                                "noise", "maintenance", "pests", "parking", "security", "appliances",
                                "plumbing", "heating_cooling", "neighbors", "management", "price",
                                "amenities", "space", "utilities", "none"
                            ]
                        )
                    ),
                    "positive_aspects": datasets.Sequence(
                        datasets.ClassLabel(
                            names=[
                                "location", "value", "space", "amenities", "quiet", "neighbors",
                                "maintenance", "management", "parking", "security", "appliances",
                                "utilities", "heating_cooling", "none"
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            task_templates=[
                TabularClassification(
                    target_column="sentiment_category",
                    input_columns=[
                        "review_text", "pros_text", "cons_text", "overall_rating",
                        "value_for_money_rating", "location_rating", "property_condition_rating",
                        "noise_level_rating", "safety_rating", "cleanliness_rating",
                        "landlord_rating", "management_rating", "maintenance_rating"
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
                    "filepath": data_dir["reviews_dataset"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["reviews_dataset"],
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["reviews_dataset"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples from the CSV files."""
        df = pd.read_csv(filepath)
        
        # Create train/validation/test splits based on review dates
        # Sort by date first to ensure chronological splitting
        df['review_date'] = pd.to_datetime(df['review_date'])
        df = df.sort_values('review_date')
        
        # Use chronological split: earliest 70% for training, next 15% for validation, latest 15% for test
        total_rows = len(df)
        train_end = int(total_rows * 0.7)
        val_end = int(total_rows * 0.85)
        
        if split == "train":
            df = df.iloc[:train_end]
        elif split == "validation":
            df = df.iloc[train_end:val_end]
        elif split == "test":
            df = df.iloc[val_end:]
        
        for id_, row in df.iterrows():
            # Convert issue categories and positive aspects from comma-separated strings to lists
            issue_cats = row["issue_categories"].split(",") if not pd.isna(row["issue_categories"]) else ["none"]
            positive_aspects = row["positive_aspects"].split(",") if not pd.isna(row["positive_aspects"]) else ["none"]
            
            yield id_, {
                "review_id": str(row["review_id"]),
                "property_id": str(row["property_id"]),
                "property_type": row["property_type"],
                "review_date": row["review_date"].strftime("%Y-%m-%d"),
                "length_of_stay_months": int(row["length_of_stay_months"]),
                "rent_amount": float(row["rent_amount"]),
                # Overall ratings
                "overall_rating": int(row["overall_rating"]),
                "would_recommend": bool(row["would_recommend"]),
                "would_rent_again": bool(row["would_rent_again"]),
                # Specific aspect ratings
                "value_for_money_rating": int(row["value_for_money_rating"]),
                "location_rating": int(row["location_rating"]),
                "property_condition_rating": int(row["property_condition_rating"]),
                "noise_level_rating": int(row["noise_level_rating"]),
                "safety_rating": int(row["safety_rating"]),
                "cleanliness_rating": int(row["cleanliness_rating"]),
                "landlord_rating": int(row["landlord_rating"]),
                "management_rating": int(row["management_rating"]),
                "maintenance_rating": int(row["maintenance_rating"]),
                # Review text and analysis
                "review_text": row["review_text"],
                "review_title": row["review_title"],
                "review_upvotes": int(row["review_upvotes"]),
                "review_downvotes": int(row["review_downvotes"]),
                "sentiment_score": float(row["sentiment_score"]),
                "sentiment_category": row["sentiment_category"],
                # Additional features
                "pros_text": row["pros_text"],
                "cons_text": row["cons_text"],
                "has_photos": bool(row["has_photos"]),
                "verified_tenant": bool(row["verified_tenant"]),
                # Text analysis features
                "issue_categories": issue_cats,
                "positive_aspects": positive_aspects,
            }