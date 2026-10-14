"""
Agronomic Disease Risk Estimation Model.
Evaluates microclimate conditions (temperature, humidity, soil moisture, leaf wetness)
to compute an Environmental Disease-Risk Index for Jamun foliage.

DISCLAIMER:
Environmental favorability alone does NOT constitute diagnostic proof of infection.
This rule-based index is an agronomic advisory tool indicating whether current
microclimate parameters create conducive conditions for pathogen sporulation and infection.
"""

from typing import Dict, Any, Tuple


class AgronomicRiskModel:
    """
    Transparent, rule-based agronomic microclimate risk indicator for Jamun diseases:
    - Anthracnose (Colletotrichum gloeosporioides)
    - Leaf Spot (Pseudocercospora / Cercospora / Pestalotiopsis spp.)
    """

    def __init__(
        self,
        anthracnose_temp_range: Tuple[float, float] = (24.0, 32.0),
        anthracnose_humidity_threshold: float = 80.0,
        anthracnose_leaf_wetness_threshold: float = 6.0,
        leaf_spot_temp_range: Tuple[float, float] = (20.0, 30.0),
        leaf_spot_humidity_threshold: float = 75.0,
        leaf_spot_leaf_wetness_threshold: float = 4.0,
    ):
        self.anthracnose_temp_range = anthracnose_temp_range
        self.anthracnose_humidity_threshold = anthracnose_humidity_threshold
        self.anthracnose_leaf_wetness_threshold = anthracnose_leaf_wetness_threshold

        self.leaf_spot_temp_range = leaf_spot_temp_range
        self.leaf_spot_humidity_threshold = leaf_spot_humidity_threshold
        self.leaf_spot_leaf_wetness_threshold = leaf_spot_leaf_wetness_threshold

    def evaluate_risk(self, sensor_readings: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate environmental risk score from current microclimate measurements.
        
        Args:
            sensor_readings: Dict containing keys:
                - ambient_temperature (°C)
                - soil_temperature (°C)
                - humidity (%)
                - soil_moisture (%)
                - leaf_wetness (hours)
                
        Returns:
            Dictionary with risk indices, risk category, pathogen breakdown, and advisory.
        """
        t_amb = sensor_readings.get("ambient_temperature", 25.0)
        t_soil = sensor_readings.get("soil_temperature", 24.0)
        rh = sensor_readings.get("humidity", 60.0)
        sm = sensor_readings.get("soil_moisture", 40.0)
        lw = sensor_readings.get("leaf_wetness", 0.0)

        # 1. Anthracnose Favorability Score (0.0 to 1.0)
        # Temperature factor (Gaussian-like score around 28°C)
        t_min, t_max = self.anthracnose_temp_range
        if t_min <= t_amb <= t_max:
            t_score_anth = 1.0 - (abs(t_amb - 28.0) / 8.0)
        else:
            t_score_anth = max(0.0, 1.0 - (min(abs(t_amb - t_min), abs(t_amb - t_max)) / 10.0))

        # Humidity factor (sigmoidal ramp above 75%)
        rh_score_anth = max(0.0, min(1.0, (rh - 60.0) / 35.0))

        # Leaf wetness factor (critical for spore germination; 0 below 2h, ramp to 1 at 8h)
        lw_score_anth = max(0.0, min(1.0, lw / 8.0))

        # Combined Anthracnose index (weighted: leaf wetness is primary driver in literature)
        anthracnose_risk = (0.25 * t_score_anth) + (0.35 * rh_score_anth) + (0.40 * lw_score_anth)

        # 2. Leaf Spot Favorability Score (0.0 to 1.0)
        ls_min, ls_max = self.leaf_spot_temp_range
        if ls_min <= t_amb <= ls_max:
            t_score_ls = 1.0 - (abs(t_amb - 26.0) / 8.0)
        else:
            t_score_ls = max(0.0, 1.0 - (min(abs(t_amb - ls_min), abs(t_amb - ls_max)) / 10.0))

        rh_score_ls = max(0.0, min(1.0, (rh - 55.0) / 35.0))
        lw_score_ls = max(0.0, min(1.0, lw / 6.0))

        leaf_spot_risk = (0.25 * t_score_ls) + (0.35 * rh_score_ls) + (0.40 * lw_score_ls)

        # Overall microclimate disease favorability
        composite_risk = max(anthracnose_risk, leaf_spot_risk)
        risk_percent = round(float(composite_risk * 100.0), 1)

        if risk_percent < 35.0:
            category = "LOW"
            advisory = "Environmental parameters are unfavorable for fungal spore germination and canopy spread."
            color = "#10b981"  # Emerald green
        elif risk_percent < 65.0:
            category = "MODERATE"
            advisory = "Moderate ambient humidity and leaf wetness detected. Continue regular canopy monitoring."
            color = "#f59e0b"  # Amber
        else:
            category = "HIGH"
            advisory = "High ambient humidity and prolonged leaf wetness create conducive conditions for pathogen sporulation."
            color = "#ef4444"  # Red

        factor_alerts = []
        if lw >= self.anthracnose_leaf_wetness_threshold:
            factor_alerts.append(f"Extended leaf wetness ({lw:.1f}h exceeds {self.anthracnose_leaf_wetness_threshold:.1f}h threshold)")
        if rh >= self.anthracnose_humidity_threshold:
            factor_alerts.append(f"Elevated relative humidity ({rh:.1f}% exceeds {self.anthracnose_humidity_threshold:.1f}%)")
        if t_min <= t_amb <= t_max:
            factor_alerts.append(f"Canopy temperature ({t_amb:.1f}°C) is within optimal fungal growth window ({t_min}-{t_max}°C)")

        return {
            "composite_risk_score": risk_percent,
            "risk_category": category,
            "category_color": color,
            "pathogen_breakdown": {
                "Anthracnose": round(float(anthracnose_risk * 100.0), 1),
                "Leaf Spot": round(float(leaf_spot_risk * 100.0), 1),
            },
            "factor_alerts": factor_alerts,
            "advisory": advisory,
            "disclaimer": (
                "Agronomic advisory notice: High environmental favorability indicates microclimate "
                "risk for spore germination, NOT independent diagnostic proof that a leaf is currently infected."
            ),
        }
