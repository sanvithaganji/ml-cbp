# Dataset Integration Guide: Jamun Plant Disease Dataset

## Dataset Origin and Description
This project is designed to evaluate:
**Jamun Plant Disease Dataset collected at DA-IICT (Dhirubhai Ambani Institute of Information and Communication Technology), Gandhinagar, Gujarat, India.**

### Expected Dataset Specifications
- **Host Crop:** Jamun (*Syzygium cumini*)
- **Total Images:** 4,195 high-resolution leaf RGB photographs
- **Class Distribution:**
  - `Anthracnose` (*Colletotrichum gloeosporioides*): 2,389 images (~56.9%)
  - `Healthy` (Asymptomatic foliage): 1,148 images (~27.4%)
  - `Leaf Spot` (*Pseudocercospora* / *Cercospora* / *Pestalotiopsis* spp.): 658 images (~15.7%)
- **Target Microclimate Sensors:**
  1. `ambient_temperature` (°C): Air temperature around canopy
  2. `soil_temperature` (°C): Root zone temperature
  3. `humidity` (%): Canopy relative humidity
  4. `soil_moisture` (%): Volumetric soil moisture content
  5. `leaf_wetness` (hours): Duration of free surface water on leaf cuticle

## Directory Layout for Real Dataset
Place the raw files under the `data/` directory using the following structure:

```
data/
├── README.md
├── metadata.csv           <-- Primary metadata mapping images to sensors & labels
├── images/                <-- Leaf images referenced by metadata.csv
│   ├── anthracnose/
│   │   ├── ANTH_0001.jpg
│   │   └── ...
│   ├── healthy/
│   │   ├── HLTH_0001.jpg
│   │   └── ...
│   └── leaf_spot/
│       ├── SPOT_0001.jpg
│       └── ...
```

## Expected Metadata Schema (`metadata.csv`)
The CSV file must contain the following columns:
| Column Name | Type | Description |
|---|---|---|
| `image_path` | String | Relative path to image, e.g. `images/anthracnose/ANTH_0001.jpg` |
| `label` | String | Exactly one of: `Healthy`, `Anthracnose`, `Leaf Spot` |
| `ambient_temperature` | Float | Ambient air temperature in °C |
| `soil_temperature` | Float | Soil temperature in °C |
| `humidity` | Float | Relative humidity in % |
| `soil_moisture` | Float | Volumetric soil moisture in % |
| `leaf_wetness` | Float | Leaf wetness duration in hours |
| `plant_id` | String (optional) | Plant identifier for grouping to prevent data leakage |
| `timestamp` | String (optional) | Acquisition timestamp |
| `stage` | String (optional) | `early`, `moderate`, `severe`, or `asymptomatic` |

## Synthetic Demo Dataset Generator
If the private DA-IICT dataset is not present in your local workspace, use the built-in testing generator:
```bash
python scripts/generate_demo_data.py --samples 300
```
> **Notice**: Synthetic data is strictly for software testing, automated CI/CD verification, and interface testing. It is clearly labeled as synthetic in all logs and must never be cited as real agronomic performance.
