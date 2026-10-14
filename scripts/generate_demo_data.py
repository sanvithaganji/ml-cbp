"""
Synthetic Demo Dataset Generator.
FOR SOFTWARE TESTING AND LOCAL PIPELINE VERIFICATION ONLY.

Generates realistic procedural Jamun leaf images with disease symptoms
and paired environmental sensor telemetry, matching the DA-IICT class distribution.
DISCLAIMER: Outputs are explicitly tagged as synthetic demo data and must NEVER
be reported as real-world agronomic performance.
"""

import os
import argparse
import random
import logging
from typing import Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CLASS_WEIGHTS = {
    "Anthracnose": 2389 / 4195,  # ~56.9%
    "Healthy": 1148 / 4195,      # ~27.4%
    "Leaf Spot": 658 / 4195,     # ~15.7%
}


def draw_jamun_leaf_base(draw: ImageDraw.ImageDraw, width: int, height: int, rng: random.Random) -> None:
    """Draw a base Jamun (Syzygium cumini) leaf outline, foliage body, and venation."""
    center_x = width // 2
    center_y = height // 2
    rx = width // 3 + rng.randint(-10, 10)
    ry = height // 2 - 20 + rng.randint(-5, 5)

    # Base foliage color (lush deep green)
    base_color = (
        rng.randint(30, 50),
        rng.randint(90, 130),
        rng.randint(30, 55),
    )

    # Draw elliptical leaf body
    bbox = [center_x - rx, center_y - ry, center_x + rx, center_y + ry]
    draw.ellipse(bbox, fill=base_color, outline=(25, 75, 25), width=2)

    # Main vein (midrib)
    midrib_color = (
        min(255, base_color[0] + 30),
        min(255, base_color[1] + 25),
        min(255, base_color[2] + 20),
    )
    draw.line([(center_x, center_y - ry + 10), (center_x, center_y + ry - 10)], fill=midrib_color, width=4)

    # Secondary lateral veins
    for y_offset in range(-ry + 30, ry - 30, 22):
        y_pos = center_y + y_offset
        dx = int(rx * (1.0 - (abs(y_offset) / ry) ** 2) ** 0.5) * 0.8
        draw.line([(center_x, y_pos), (center_x - dx, y_pos - 12)], fill=midrib_color, width=2)
        draw.line([(center_x, y_pos), (center_x + dx, y_pos - 12)], fill=midrib_color, width=2)


def render_synthetic_leaf(label: str, size: int = 224, seed: int = 42) -> Image.Image:
    """
    Generate procedural RGB leaf image with class-specific morphological features.
    """
    rng = random.Random(seed)
    # Neutral agronomic background (light cream / gray bench)
    bg_color = (rng.randint(230, 245), rng.randint(230, 245), rng.randint(225, 240))
    img = Image.new("RGB", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)

    draw_jamun_leaf_base(draw, size, size, rng)

    center_x = size // 2
    center_y = size // 2

    if label == "Anthracnose":
        # Irregular necrotic dark brown lesions with chlorotic yellowish borders
        num_lesions = rng.randint(5, 12)
        for _ in range(num_lesions):
            lx = center_x + rng.randint(-size // 4, size // 4)
            ly = center_y + rng.randint(-size // 3, size // 3)
            r = rng.randint(12, 28)

            # Outer chlorotic yellow halo
            draw.ellipse(
                [lx - r - 4, ly - r - 4, lx + r + 4, ly + r + 4],
                fill=(rng.randint(170, 200), rng.randint(150, 180), 40),
            )
            # Inner dark necrotic blotch
            draw.ellipse(
                [lx - r, ly - r, lx + r, ly + r],
                fill=(rng.randint(45, 70), rng.randint(25, 45), rng.randint(15, 30)),
            )

    elif label == "Leaf Spot":
        # Defined circular target-like spots with reddish-brown concentric edges
        num_spots = rng.randint(8, 20)
        for _ in range(num_spots):
            lx = center_x + rng.randint(-size // 4, size // 4)
            ly = center_y + rng.randint(-size // 3, size // 3)
            r = rng.randint(5, 12)

            # Defined circular spot
            draw.ellipse(
                [lx - r, ly - r, lx + r, ly + r],
                fill=(rng.randint(110, 140), rng.randint(40, 60), rng.randint(25, 40)),
                outline=(60, 20, 15),
                width=1,
            )
            # Center ash-gray necrotic core
            draw.ellipse(
                [lx - r // 2, ly - r // 2, lx + r // 2, ly + r // 2],
                fill=(rng.randint(140, 170), rng.randint(130, 160), rng.randint(120, 150)),
            )

    # Slight Gaussian softening for natural photographic texture
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img


def generate_paired_sensors(label: str, rng: np.random.Generator) -> dict:
    """
    Generate realistic microclimate readings conditioned on class distribution.
    Reflects agronomic environmental correlation while maintaining realistic overlap.
    """
    if label == "Anthracnose":
        # Warm, high humidity, extended leaf wetness
        t_amb = float(np.clip(rng.normal(28.5, 2.2), 22.0, 36.0))
        t_soil = float(np.clip(t_amb - rng.uniform(1.5, 3.5), 20.0, 32.0))
        humidity = float(np.clip(rng.normal(86.0, 5.0), 72.0, 98.0))
        soil_moist = float(np.clip(rng.normal(68.0, 9.0), 40.0, 95.0))
        leaf_wet = float(np.clip(rng.normal(8.5, 2.5), 4.0, 16.0))
    elif label == "Leaf Spot":
        # Moderate warm, elevated humidity, moderate leaf wetness
        t_amb = float(np.clip(rng.normal(26.0, 2.5), 19.0, 33.0))
        t_soil = float(np.clip(t_amb - rng.uniform(1.0, 3.0), 18.0, 30.0))
        humidity = float(np.clip(rng.normal(78.0, 6.0), 65.0, 92.0))
        soil_moist = float(np.clip(rng.normal(55.0, 10.0), 30.0, 85.0))
        leaf_wet = float(np.clip(rng.normal(5.5, 2.0), 2.0, 11.0))
    else:  # Healthy
        # Moderate temperature, lower humidity, minimal leaf wetness
        t_amb = float(np.clip(rng.normal(24.5, 3.0), 18.0, 32.0))
        t_soil = float(np.clip(t_amb - rng.uniform(0.5, 2.5), 17.0, 29.0))
        humidity = float(np.clip(rng.normal(52.0, 8.0), 30.0, 68.0))
        soil_moist = float(np.clip(rng.normal(42.0, 8.0), 20.0, 65.0))
        leaf_wet = float(np.clip(rng.exponential(0.8), 0.0, 3.0))

    return {
        "ambient_temperature": round(t_amb, 2),
        "soil_temperature": round(t_soil, 2),
        "humidity": round(humidity, 2),
        "soil_moisture": round(soil_moist, 2),
        "leaf_wetness": round(leaf_wet, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic demo dataset for software verification.")
    parser.add_argument("--samples", type=int, default=300, help="Total sample count to generate (default: 300).")
    parser.add_argument("--output_dir", type=str, default="data", help="Output data directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    np_rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    images_base_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_base_dir, exist_ok=True)

    for cname in ["healthy", "anthracnose", "leaf_spot"]:
        os.makedirs(os.path.join(images_base_dir, cname), exist_ok=True)

    # Determine counts matching DA-IICT distribution
    classes = ["Anthracnose", "Healthy", "Leaf Spot"]
    probs = [CLASS_WEIGHTS[c] for c in classes]
    sampled_classes = np_rng.choice(classes, size=args.samples, p=probs)

    records = []
    # Assign to 20 synthetic plant IDs to enable group-based leak-free splitting
    num_plants = max(10, args.samples // 15)

    logger.info("Generating %d synthetic demo samples across %d plant groups...", args.samples, num_plants)

    for idx, label in enumerate(sampled_classes):
        plant_idx = idx % num_plants
        plant_id = f"PLANT_{plant_idx:03d}"
        folder_name = label.lower().replace(" ", "_")
        filename = f"DEMO_{label[:4].upper()}_{idx:04d}.jpg"
        rel_img_path = os.path.join("images", folder_name, filename)
        abs_img_path = os.path.join(args.output_dir, rel_img_path)

        # Generate and save procedural leaf image
        img = render_synthetic_leaf(label=label, size=224, seed=args.seed + idx)
        img.save(abs_img_path, quality=92)

        # Generate paired microclimate sensors
        sensors = generate_paired_sensors(label=label, rng=np_rng)

        records.append({
            "image_path": rel_img_path,
            "label": label,
            **sensors,
            "plant_id": plant_id,
            "timestamp": f"2026-10-13T{10 + (idx % 12):02d}:{(idx * 7) % 60:02d}:00Z",
            "dataset_origin": "SYNTHETIC_DEMO_FOR_SOFTWARE_TESTING_ONLY",
        })

    metadata_df = pd.DataFrame(records)
    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)

    logger.info("Successfully generated %d demo images at %s", len(metadata_df), images_base_dir)
    logger.info("Saved metadata CSV at %s", metadata_path)
    logger.info("Class distribution:\n%s", metadata_df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
