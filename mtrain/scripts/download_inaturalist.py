#!/usr/bin/env python3
"""
Script to download data from iNaturalist based on search parameters.
Targets 1500 samples from plant observations in India with research grade quality.
Uses pyinaturalist SDK for API interaction.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import requests
from pyinaturalist import get_observations
from tqdm import tqdm


class INaturalistDownloader:
    """Downloads observations from iNaturalist using pyinaturalist SDK."""

    def __init__(
        self, output_dir: str = "inaturalist_data", rate_limit_delay: float = 0.5
    ):
        self.output_dir = Path(output_dir)
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "iNaturalist-Research-Download/1.0 (research purposes)"}
        )

    def get_observations(
        self,
        place_id: int = 6681,  # India
        taxon_id: int = 47125,  # Plantae
        quality_grade: str = "research",
        photos: bool = True,
        per_page: int = 200,
        max_results: int = 1500,
    ) -> List[Dict]:
        """Fetch observations using pyinaturalist SDK."""

        observations = []
        page = 1

        print(f"Fetching observations with pyinaturalist SDK...")
        print(
            f"Parameters: place_id={place_id}, taxon_id={taxon_id}, quality_grade={quality_grade}"
        )

        with tqdm(total=max_results, desc="Fetching observations") as pbar:
            while len(observations) < max_results:
                try:
                    # Use pyinaturalist to get observations
                    response = get_observations(
                        place_id=place_id,
                        taxon_id=taxon_id,
                        quality_grade=quality_grade,
                        photos=photos,
                        per_page=per_page,
                        page=page,
                        order="desc",
                        order_by="created_at",
                    )

                    page_observations = response.get("results", [])

                    if not page_observations:
                        print(f"No more results found at page {page}")
                        break

                    observations.extend(page_observations)
                    pbar.update(len(page_observations))

                    if len(page_observations) < per_page:
                        print(f"Reached end of available data at page {page}")
                        break

                    page += 1

                except Exception as e:
                    print(f"Error fetching page {page}: {e}")
                    break

        return observations[:max_results]

    def download_photos(
        self, observations: List[Dict], max_photos_per_obs: int = 3
    ) -> None:
        """Download photos for the given observations."""

        photos_dir = self.output_dir / "photos"
        photos_dir.mkdir(parents=True, exist_ok=True)

        total_photos = sum(
            min(len(obs.get("photos", [])), max_photos_per_obs) for obs in observations
        )

        with tqdm(total=total_photos, desc="Downloading photos") as pbar:
            for obs in observations:
                obs_id = obs["id"]
                photos = obs.get("photos", [])[:max_photos_per_obs]

                for i, photo in enumerate(photos):
                    try:
                        # Get the large size URL for 1024x1024 images
                        photo_url = photo.get("url")
                        if photo_url is not None:
                            photo_url = photo_url.replace("square", "large")
                        else:
                            for size in ["large", "medium", "small"]:
                                if size in photo and photo[size]:
                                    photo_url = photo[size]
                                    break

                        if not photo_url:
                            continue

                        # Determine file extension
                        ext = photo_url.split(".")[-1].split("?")[0].lower()
                        if ext not in ["jpg", "jpeg", "png"]:
                            ext = "jpg"

                        filename = f"{obs_id}_{i}.{ext}"
                        filepath = photos_dir / filename

                        if filepath.exists():
                            print("skip: file exists; {filepath}")
                            continue

                        # Download photo
                        response = self.session.get(photo_url, stream=True)
                        response.raise_for_status()

                        # ext = photo_url.split(".")[-1].split("?")[0].lower()
                        # if ext not in ["jpg", "jpeg", "png"]:
                        #     ext = "jpg"

                        # filename = f"{obs_id}_{i}.{ext}"
                        # filepath = photos_dir / filename

                        with open(filepath, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)

                        pbar.update(1)
                        time.sleep(self.rate_limit_delay)

                    except requests.RequestException as e:
                        print(f"Error downloading photo {photo_url}: {e}")
                        pbar.update(1)
                        continue

    def save_metadata(self, observations: List[Dict]) -> None:
        """Save observation metadata as JSON."""

        metadata_file = self.output_dir / "metadata.json"

        # Clean up observations to keep only essential data
        clean_observations = []
        for obs in observations:
            clean_obs = {
                "id": obs["id"],
                "location": obs.get("location"),
                "place_ids": obs.get("place_ids", []),
                "taxon": {
                    "id": obs.get("taxon", {}).get("id"),
                    "name": obs.get("taxon", {}).get("name"),
                    "preferred_common_name": obs.get("taxon", {}).get(
                        "preferred_common_name"
                    ),
                    "rank": obs.get("taxon", {}).get("rank"),
                    "ancestry": obs.get("taxon", {}).get("ancestry"),
                },
                "quality_grade": obs.get("quality_grade"),
                "photos": [
                    {
                        "id": photo.get("id"),
                        "medium": photo.get("medium"),
                        "large": photo.get("large"),
                    }
                    for photo in obs.get("photos", [])
                ],
                "user": {
                    "id": obs.get("user", {}).get("id"),
                    "login": obs.get("user", {}).get("login"),
                },
            }
            clean_observations.append(clean_obs)

        print(clean_observations)
        with open(metadata_file, "w") as f:
            json.dump(clean_observations, f, indent=2)

        print(
            f"Saved metadata for {len(clean_observations)} observations to {metadata_file}"
        )

    def create_dataset_summary(self, observations: List[Dict]) -> None:
        """Create a summary of the downloaded dataset."""

        summary_file = self.output_dir / "dataset_summary.txt"

        # Count species
        species_counts = {}
        quality_counts = {}

        for obs in observations:
            taxon = obs.get("taxon", {})
            species_name = taxon.get("name", "Unknown")
            species_counts[species_name] = species_counts.get(species_name, 0) + 1

            quality = obs.get("quality_grade", "unknown")
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        with open(summary_file, "w") as f:
            f.write(f"iNaturalist Dataset Summary\n")
            f.write(f"==========================\n\n")
            f.write(f"Total observations: {len(observations)}\n")
            f.write(f"Total species: {len(species_counts)}\n\n")

            f.write("Quality Grade Distribution:\n")
            for grade, count in sorted(quality_counts.items()):
                f.write(f"  {grade}: {count}\n")

            f.write(f"\nTop 20 Species:\n")
            sorted_species = sorted(
                species_counts.items(), key=lambda x: x[1], reverse=True
            )
            for i, (species, count) in enumerate(sorted_species[:20], 1):
                f.write(f"  {i:2d}. {species}: {count}\n")

            if len(sorted_species) > 20:
                f.write(f"  ... and {len(sorted_species) - 20} more species\n")

        print(f"Created dataset summary at {summary_file}")

    def download(
        self,
        place_id: int = 6681,
        taxon_id: int = 47125,
        quality_grade: str = "research",
        max_results: int = 1500,
        download_photos: bool = True,
        max_photos_per_obs: int = 3,
    ) -> None:
        """Main download function."""

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting iNaturalist download...")
        print(f"Place ID: {place_id}")
        print(f"Taxon ID: {taxon_id}")
        print(f"Quality grade: {quality_grade}")
        print(f"Target samples: {max_results}")
        print(f"Output directory: {self.output_dir}")
        print(f"Download photos: {download_photos}")
        print(f"Max photos per observation: {max_photos_per_obs}")

        # Fetch observations
        observations = self.get_observations(
            place_id=place_id,
            taxon_id=taxon_id,
            quality_grade=quality_grade,
            max_results=max_results,
        )

        if not observations:
            print("No observations found!")
            return

        print(f"Found {len(observations)} observations")

        # Save metadata
        self.save_metadata(observations)

        # Download photos if requested
        if download_photos:
            print("Downloading photos...")
            self.download_photos(observations, max_photos_per_obs)

        # Create summary
        self.create_dataset_summary(observations)

        print(f"\nDownload complete! Data saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download iNaturalist observations")
    parser.add_argument(
        "--output-dir",
        default="inaturalist_data",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--place-id", type=int, default=6681, help="Place ID (default: 6681 for India)"
    )
    parser.add_argument(
        "--taxon-id",
        type=int,
        default=47125,
        help="Taxon ID (default: 47125 for Plantae)",
    )
    parser.add_argument(
        "--quality-grade",
        default="research",
        choices=["research", "needs_id", "casual"],
        help="Quality grade filter",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=1500,
        help="Maximum number of observations to download",
    )
    parser.add_argument(
        "--no-photos", action="store_true", help="Skip photo downloads (metadata only)"
    )
    parser.add_argument(
        "--max-photos-per-obs",
        type=int,
        default=3,
        help="Maximum photos per observation",
    )
    parser.add_argument(
        "--rate-limit", type=float, default=0.5, help="Delay between requests (seconds)"
    )

    args = parser.parse_args()

    downloader = INaturalistDownloader(
        output_dir=args.output_dir, rate_limit_delay=args.rate_limit
    )

    downloader.download(
        place_id=args.place_id,
        taxon_id=args.taxon_id,
        quality_grade=args.quality_grade,
        max_results=args.max_results,
        download_photos=not args.no_photos,
        max_photos_per_obs=args.max_photos_per_obs,
    )


if __name__ == "__main__":
    main()


# [
#     {
#         "id": 623569536,
#         "license_code": None,
#         "original_dimensions": {"width": 2048, "height": 1536},
#         "url": "https://static.inaturalist.org/photos/623569536/square.jpg",
#         "attribution": "(c) Pavan Patel, all rights reserved",
#         "flags": [],
#         "moderator_actions": [],
#         "hidden": False,
#     }
# ]
