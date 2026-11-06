"""Prompt/retrieval functions preserved from the submitted-study source.

Only template construction and retrieval are included; no model execution.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


MODE_STR_TO_INT = {
    'public transport': 3, 'public transit': 3, 'transit': 3,
    'bus': 3, 'train': 3,
    'walk': 1, 'walking': 1,
    'cycle': 2, 'cycling': 2, 'bicycle': 2,
    'drive': 4, 'driving': 4, 'car': 4,
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

def trip_to_text(row, include_label: bool = True) -> str:
    t = get_traveller_info(row)

    text = (
        f"Profile: Age {row['age']}, {t['gender']}, {t['license_str']}, {t['car_str']}.\n"
        f"Trip: {t['purpose_str']}, {row['distance']:.0f} m, departs {row['start_time']:.1f}:00.\n"
        f"Options - walk: {row['dur_walking']:.2f}h £0.00 | "
        f"cycle: {row['dur_cycling']:.2f}h £0.00 | "
        f"public transport: {t['pt_dur']:.2f}h £{row['cost_transit']:.2f} | "
        f"drive: {row['dur_driving']:.2f}h £{t['drive_cost']:.2f}"
    )

    if include_label:
        text += f"\nCHOICE: {MODE_INT_TO_STR[int(row['travel_mode'])]}"

    return text

def build_persona_system_msg(query_row, n_examples: int) -> str:
    t = get_traveller_info(query_row)
    age = int(query_row['age'])

    if age < 25:
        life_stage = "a young adult"
    elif age < 40:
        life_stage = "in your prime working years"
    elif age < 55:
        life_stage = "middle-aged"
    elif age < 65:
        life_stage = "approaching retirement"
    else:
        life_stage = "a senior"

    sh = query_row['start_time']

    if 6 <= sh < 10:
        time_str = "morning"
    elif 10 <= sh < 12:
        time_str = "late morning"
    elif 12 <= sh < 14:
        time_str = "midday"
    elif 14 <= sh < 18:
        time_str = "afternoon"
    elif 18 <= sh < 22:
        time_str = "evening"
    else:
        time_str = "late night or early morning"

    persona = (
        f"You are a London resident, {age} years old, {t['gender']}, {life_stage}. "
        f"You {t['license_str']} and you have {t['car_str']}. "
        f"It is {time_str} ({sh:.1f}:00), and you are about to start a trip - "
        f"{t['purpose_str']}, covering {query_row['distance']:.0f} metres. "
        f"You have four options: walk, cycle, public transport, or drive."
    )

    return (
        "You are simulating a real London traveller making an everyday travel decision. "
        "Fully adopt the identity described below - think, feel, and reason as this specific "
        "person would. Consider your age, gender, household situation, the time of day, the "
        "trip purpose, the distance, and what each mode would cost you in time and money.\n\n"
        f"[YOUR IDENTITY]\n{persona}\n\n"
        f"Before deciding, you will see {n_examples} example trips made by other Londoners in "
        "situations similar to yours. Learn from their choices, but make your own decision based "
        "on who YOU are and what makes sense for YOUR specific situation. Stay fully in character "
        "throughout - use 'I' when reasoning, not 'the traveller'."
    )

def build_fewshot_prompt(query_row, examples_df: pd.DataFrame) -> list:
    n_examples = len(examples_df)

    messages = [
        {
            "role": "system",
            "content": build_persona_system_msg(query_row, n_examples),
        }
    ]

    for _, ex_row in examples_df.iterrows():
        messages.append({
            "role": "user",
            "content": (
                "Here is an example trip made by another Londoner:\n\n"
                + trip_to_text(ex_row, include_label=False)
            ),
        })

        messages.append({
            "role": "assistant",
            "content": f"CHOICE: {MODE_INT_TO_STR[int(ex_row['travel_mode'])]}",
        })

    t = get_traveller_info(query_row)

    query_text = (
        f"Now it is your turn to decide. Remember, you ARE this person:\n\n"
        f"- I am {query_row['age']} years old, {t['gender']}\n"
        f"- I {t['license_str']}\n"
        f"- I have {t['car_str']}\n"
        f"- My trip purpose: {t['purpose_str']}\n"
        f"- Departure time: {query_row['start_time']:.1f}:00\n\n"
        f"My available options for this trip of {query_row['distance']:.0f} metres:\n"
        f"- walk:             {query_row['dur_walking']:.2f} hours,  £0.00\n"
        f"- cycle:            {query_row['dur_cycling']:.2f} hours,  £0.00\n"
        f"- public transport: {t['pt_dur']:.2f} hours,  £{query_row['cost_transit']:.2f}\n"
        f"- drive:            {query_row['dur_driving']:.2f} hours,  £{t['drive_cost']:.2f}\n\n"
        f"As this person, decide what YOU would actually choose. Think about your age, your "
        f"physical comfort, your time, your wallet, the convenience for your trip purpose.\n\n"
        f"First give a brief reason in first person, for example \"I would choose ... because I ...\", "
        f"then reply with ONLY one of these exact words: walk, cycle, public transport, drive.\n"
        f"Use this exact format: \"REASON: [your reason here]. CHOICE: [walk/cycle/public transport/drive]\".\n"
    )

    messages.append({
        "role": "user",
        "content": query_text,
    })

    return messages
