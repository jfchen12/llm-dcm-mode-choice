"""Prompt/retrieval functions preserved from the submitted-study source.

Only template construction and retrieval are included; no model execution.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


MODE_STR_TO_INT = {
    'public transport': 3, 'public transit': 3, 'transit': 3,
    'bus': 3, 'train': 3,
    'walking': 1, 'walk': 1,
    'cycling': 2, 'cycle': 2, 'bicycle': 2,
    'driving': 4, 'drive': 4, 'car': 4,
}

MODE_INT_TO_STR = {
    1: 'walk',
    2: 'cycle',
    3: 'public transport',
    4: 'drive',
}

MODES = [1, 2, 3, 4]

def get_traveller_info(row):
    gender = "female" if row['female'] == 1 else "male"

    license_str = (
        "have a driving licence"
        if row['driving_license'] == 1
        else "do not have a driving licence"
    )

    car_map = {
        0: "no car in my household",
        1: "less than one car per adult in my household",
        2: "one or more cars per adult in my household",
    }

    purpose_map = {
        1: "commuting to work",
        2: "going to school/education",
        3: "other home-based trip",
        4: "employer's business",
        5: "other non-home-based trip",
    }

    car_str = car_map.get(int(row['car_ownership']), "unknown")
    purpose_str = purpose_map.get(int(row['purpose']), "a trip")

    pt_dur = (
        row['dur_pt_access']
        + row['dur_pt_rail']
        + row['dur_pt_bus']
        + row['dur_pt_int']
    )

    drive_cost = row['cost_driving_fuel'] + row['cost_driving_ccharge']

    return dict(
        gender=gender,
        license_str=license_str,
        car_str=car_str,
        purpose_str=purpose_str,
        pt_dur=pt_dur,
        drive_cost=drive_cost,
    )

def build_prompt(row):
    t = get_traveller_info(row)

    return f"""Predict the most likely travel mode for this traveller.

Your profile:
- Age: {row['age']}, {t['gender']}
- You {t['license_str']}
- You have {t['car_str']}
- Trip purpose: {t['purpose_str']}
- Departure time: {row['start_time']:.1f}:00

Your available options for this trip of {row['distance']:.0f} metres:
- walk:             {row['dur_walking']:.2f} hours,  £0.00
- cycle:            {row['dur_cycling']:.2f} hours,  £0.00
- public transport: {t['pt_dur']:.2f} hours,  £{row['cost_transit']:.2f}
- drive:            {row['dur_driving']:.2f} hours,  £{t['drive_cost']:.2f}

First give a brief reason why you choose that mode, then reply with ONLY one of these exact words: walk, cycle, public transport, drive.
Use this exact format: "REASON: [your reason here]. CHOICE: [walk/cycle/public transport/drive]".

"""
