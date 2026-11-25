"""Water-chemistry terms for the pond twin. Pure functions over (state, config, forcing).

CALIBRATION HONESTY: every coefficient lives in ``configs/twin.yaml`` and is a *calibratable
default*, not a measured fact. The functional forms follow standard pond-aquaculture practice
(Boyd, *Water Quality in Ponds for Aquaculture*; FAO pond manuals; APHA Standard Methods for
DO saturation), but the numbers were chosen so a warm-water earthen pond behaves plausibly -
dawn DO sag, daytime pH rise, ammonia build-up after feeding. Calibrate before trusting any
output as physical.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


# ----------------------------------------------------------------------------------------
# Forcing
# ----------------------------------------------------------------------------------------
def light_fraction(hour_of_day: float, light_cfg: Dict[str, Any], cloud: float = 1.0) -> float:
    """Incident-light fraction in [0, 1]: a half-sine between sunrise and sunset, else 0.

    Photosynthesis is exactly zero at night - that is what produces the dawn DO sag the whole
    scenario set turns on.
    """
    sunrise = float(light_cfg["sunrise_h"])
    sunset = float(light_cfg["sunset_h"])
    if not (sunrise < hour_of_day < sunset):
        return 0.0
    phase = (hour_of_day - sunrise) / (sunset - sunrise)
    return float(math.sin(math.pi * phase) * max(cloud, 0.0))


def temperature_forcing(hour_of_day: float, temp_cfg: Dict[str, Any], offset_c: float = 0.0) -> float:
    """Target water temperature: diurnal sinusoid peaking at ``peak_hour``."""
    mean = float(temp_cfg["mean_c"]) + offset_c
    amp = float(temp_cfg["amplitude_c"])
    peak = float(temp_cfg["peak_hour"])
    return mean + amp * math.cos(2.0 * math.pi * (hour_of_day - peak) / 24.0)


# ----------------------------------------------------------------------------------------
# Oxygen
# ----------------------------------------------------------------------------------------
def do_saturation(temp_c: float, salinity_ppt: float = 0.0) -> float:
    """DO at saturation, mg/L. Benson & Krause / APHA freshwater fit with a salinity term."""
    t_k = float(temp_c) + 273.15
    ln_c = (
        -139.34411
        + 1.575701e5 / t_k
        - 6.642308e7 / t_k**2
        + 1.243800e10 / t_k**3
        - 8.621949e11 / t_k**4
    )
    # Salinity correction (Benson & Krause); negligible below ~1 ppt but kept for brackish ponds.
    ln_c -= salinity_ppt * (0.017674 - 10.754 / t_k + 2140.7 / t_k**2)
    return float(math.exp(ln_c))


def arrhenius(temp_c: float, theta: float, ref_c: float = 20.0) -> float:
    """Standard theta^(T - 20) temperature multiplier used for biological rates."""
    return float(theta ** (temp_c - ref_c))


def do_rate(
    state: Dict[str, float],
    cfg: Dict[str, Any],
    light: float,
    aerator: float,
    nitrification_n: float,
) -> Dict[str, float]:
    """dDO/dt in mg/L/h, broken out by term so the dashboard can explain a crash.

        dDO/dt = k_a*(DO_sat - DO)          surface reaeration
               + P_max*light*algae          photosynthesis (zero at night)
               - R_20*theta^(T-20)*(biomass + w*algae)   respiration
               - k_bod*BOD_pool                          organic decay
               - 4.57*nitrification_N                    nitrifier oxygen demand
               + k_aer*aerator*(DO_sat - DO)             mechanical aeration
    """
    ox = cfg["oxygen"]
    temp = state["temp_c"]
    sat = do_saturation(temp, cfg["pond"]["salinity_ppt"])
    deficit = sat - state["do"]

    reaeration = float(ox["k_reaeration_per_h"]) * deficit
    aeration = float(ox["k_aerator_per_h"]) * float(aerator) * deficit
    photosynthesis = float(ox["p_max_per_h"]) * light * state["algae"]
    resp_load = state["biomass"] + float(ox["algae_respiration_weight"]) * state["algae"]
    respiration = float(ox["r20_per_h"]) * arrhenius(temp, float(ox["theta_resp"])) * resp_load
    bod = float(ox["k_bod_per_h"]) * state["bod_pool"]
    nitrifier_demand = float(ox["o2_per_n_nitrified"]) * nitrification_n

    return {
        "reaeration": reaeration,
        "aeration": aeration,
        "photosynthesis": photosynthesis,
        "respiration": -respiration,
        "bod": -bod,
        "nitrification": -nitrifier_demand,
        "total": reaeration + aeration + photosynthesis - respiration - bod - nitrifier_demand,
        "saturation": sat,
    }


# ----------------------------------------------------------------------------------------
# Nitrogen
# ----------------------------------------------------------------------------------------
def nh3_fraction(ph: float, temp_c: float) -> float:
    """Un-ionised fraction of total ammonia. Emerson et al. (1975).

        pKa = 0.09018 + 2729.92 / T_K ;  f = 1 / (1 + 10^(pKa - pH))

    NH3 is the toxic form, so this is the channel the risk bands are written against - a pond
    at TAN 1.0 is harmless at pH 7 and lethal at pH 9.
    """
    t_k = float(temp_c) + 273.15
    pka = 0.09018 + 2729.92 / t_k
    return float(1.0 / (1.0 + 10.0 ** (pka - float(ph))))


def nitrogen_rates(state: Dict[str, float], cfg: Dict[str, Any]) -> Dict[str, float]:
    """TAN and NO2 rates in mg/L/h. Nitrification is DO-limited via a Monod term."""
    am = cfg["ammonia"]
    temp = state["temp_c"]
    theta = arrhenius(temp, float(am["theta_nitrif"]))
    do_limit = state["do"] / (float(am["k_do_half_sat"]) + max(state["do"], 0.0))

    # feed_pool releases nitrogen (and oxygen demand) as it is metabolised.
    release = float(am["feed_release_per_h"]) * state["feed_pool"]
    excretion = release * float(am["feed_protein_n_frac"]) * 1000.0 / _volume(cfg)

    nitrif_tan = float(am["k_nitrification_per_h"]) * theta * do_limit * max(state["tan"], 0.0)
    uptake = float(am["k_algae_uptake_per_h"]) * state["algae"] * _monod(
        state["tan"], float(am["n_half_sat"]) if "n_half_sat" in am else 0.25
    )
    nitrite_ox = float(am["k_nitrite_ox_per_h"]) * theta * do_limit * max(state["no2"], 0.0)

    return {
        "feed_release_kg_per_h": release,
        "tan_in": excretion,
        "tan_nitrified": nitrif_tan,
        "tan_uptake": uptake,
        "d_tan": excretion - nitrif_tan - uptake,
        "d_no2": nitrif_tan - nitrite_ox,
        "no2_oxidised": nitrite_ox,
    }


def _monod(x: float, half: float) -> float:
    x = max(float(x), 0.0)
    return x / (half + x) if (half + x) > 0 else 0.0


def _volume(cfg: Dict[str, Any]) -> float:
    return float(cfg["pond"]["volume_m3"])


# ----------------------------------------------------------------------------------------
# pH / alkalinity
# ----------------------------------------------------------------------------------------
def ph_rate(
    state: Dict[str, float],
    cfg: Dict[str, Any],
    photosynthesis: float,
    respiration: float,
) -> float:
    """dpH/dt. CO2 stripped by photosynthesis raises pH; respiration lowers it.

    Buffering scales as ``alkalinity_ref / alkalinity`` - a pond whose carbonate buffer has
    collapsed swings far harder for the same biological activity, which is the ph_swing
    scenario in one line.
    """
    p = cfg["ph"]
    alk = max(state["alkalinity"], float(p["min_alkalinity"]))
    # A collapsed buffer swings harder, but not without limit - the carbonate system cannot
    # move pH arbitrarily fast, so both the gain and the resulting rate are capped.
    buffer_gain = min(float(p["alkalinity_ref"]) / alk, float(p["max_buffer_gain"]))
    drive = (
        float(p["photosynthesis_gain"]) * photosynthesis
        - float(p["respiration_gain"]) * respiration
    ) * buffer_gain
    cap = float(p["max_drive_per_h"])
    drive = max(-cap, min(cap, drive))
    relax = float(p["relax_per_h"]) * (float(p["equilibrium"]) - state["ph"])
    return float(drive + relax)


def alkalinity_rate(state: Dict[str, float], cfg: Dict[str, Any], nitrified_n: float) -> float:
    """Nitrification consumes 7.14 mg CaCO3 per mg N (Boyd), plus a background loss."""
    p = cfg["ph"]
    consumed = float(cfg["ammonia"]["alkalinity_per_n"]) * nitrified_n
    decay = float(p["alkalinity_decay_per_h"]) * state["alkalinity"]
    return float(-consumed - decay)


# ----------------------------------------------------------------------------------------
# Algae / turbidity
# ----------------------------------------------------------------------------------------
def algae_rate(
    state: Dict[str, float],
    cfg: Dict[str, Any],
    light: float,
    growth_multiplier: float = 1.0,
    death_multiplier: float = 1.0,
) -> Dict[str, float]:
    """Logistic growth on light + nitrogen, with crowding mortality driving the bloom crash."""
    a = cfg["algae"]
    algae = max(state["algae"], 0.0)
    cap = float(a["carrying_capacity"])
    n_limit = _monod(state["tan"], float(a["n_half_sat"]))
    growth = float(a["growth_per_h"]) * growth_multiplier * light * n_limit * algae * (
        1.0 - algae / cap
    )
    crowding = float(a["crowding_death_per_h"]) * (algae / cap) ** 2 * algae
    death = (float(a["death_per_h"]) * algae + crowding) * death_multiplier
    return {"growth": growth, "death": death, "d_algae": growth - death}


def turbidity_rate(state: Dict[str, float], cfg: Dict[str, Any], d_algae: float) -> float:
    t = cfg["turbidity"]
    target = float(t["base_ntu"]) + float(t["algae_gain"]) * max(state["algae"], 0.0)
    return float(float(t["settle_per_h"]) * (target - state["turbidity"]))


# ----------------------------------------------------------------------------------------
# Process noise
# ----------------------------------------------------------------------------------------
def ou_step(
    value: float,
    noise_state: float,
    sigma: float,
    theta: float,
    dt_h: float,
    rng: np.random.Generator,
) -> tuple:
    """One Ornstein-Uhlenbeck increment: mean-reverting noise, not a random walk.

    White noise on a 5-minute step looks like sensor jitter; OU noise looks like weather.
    """
    d = -theta * noise_state * dt_h + sigma * math.sqrt(dt_h) * rng.standard_normal()
    new_noise = noise_state + d
    return value + (new_noise - noise_state), new_noise
