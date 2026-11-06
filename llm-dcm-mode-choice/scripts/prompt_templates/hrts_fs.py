"""Prompt/retrieval functions preserved from the submitted-study source.

Only template construction and retrieval are included; no model execution.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


MODE_STR_TO_INT = {
    'public transport': 2, 'public transit': 2, 'transit': 2,
    'bus': 2, 'train': 2, 'pt': 2,
    'walk': 4, 'walking': 4, 'on foot': 4,
    'cycle': 3, 'cycling': 3, 'bicycle': 3, 'bike': 3,
    'drive': 1, 'driving': 1, 'car': 1,
}

MODE_INT_TO_STR = {1: 'drive', 2: 'public transport', 3: 'cycle', 4: 'walk'}

MODES = [1, 2, 3, 4]

LABEL_COL = 'choice_mode'

GENDER_MAP   = {1: 'female', 2: 'male', 3: 'another gender',          # ✓ 码本：1=Nainen/Female, 2=Mies/Male
                4: 'unspecified gender'}

DATETYPE_MAP = {'ARKI': 'a weekday', 'VKOLOPPU': 'a weekend day'}

BICYCLE_MAP  = {1: 'almost always have a bicycle available',          # ✓ polkup：bike availability
                2: 'sometimes have a bicycle available',
                3: 'rarely have a bicycle available',
                4: 'have no bicycle available', 99: 'have unknown bicycle access'}

PTFREQ_MAP   = {1: 'use public transport daily or almost daily',      # ✓ mu_jouk：使用频率
                2: 'use public transport a few times a week',
                3: 'use public transport about once a week',
                4: 'use public transport less than once a week',
                5: 'never use public transport', 99: 'have unknown public-transport usage'}

OCCUPATION_MAP = {1: 'employed', 2: 'unemployed', 3: 'a student',     # ✓ toimi
                  4: 'a pensioner', 5: 'on parental leave',
                  6: 'in military/civilian service',
                  7: 'out of the labour force', 99: 'of unknown occupation'}

EDUCATION_MAP  = {1: 'elementary school', 2: 'comprehensive/middle school',   # ✓ tutkinto
                  3: 'an upper-secondary or vocational qualification',
                  4: 'a university-of-applied-sciences degree',
                  5: "a bachelor's degree", 6: "a master's degree",
                  7: 'a doctoral/licentiate degree', 99: 'an unknown level of education'}

HOUSING_MAP    = {1: 'an apartment building', 2: 'a terraced/semi-detached house',  # ✓ asumism
                  3: 'a detached house', 4: 'another type of housing',
                  99: 'unknown housing'}

CARAVAIL_MAP   = {1: 'a car is almost always available to you',       # ✓ mu_autok：个人可用车频率
                  2: 'a car is occasionally available to you',
                  3: 'a car is rarely available to you',
                  4: 'no car is available to you', 99: 'car availability is unknown'}

PLACE_MAP = {1: 'home', 2: 'workplace', 3: 'school/place of study',
             4: 'childcare/school drop-off', 5: 'a shop or errand', 6: 'work-related business',
             7: 'visiting friends/relatives', 8: 'exercise/outdoor recreation',
             9: 'a leisure location', 10: 'another location'}

def get_traveller_info(row):
    gender = GENDER_MAP.get(int(row['Gender']), 'unspecified gender')

    dl = row['HasDrivingLicense']
    if dl == 1:
        license_str = "have a driving licence"
    elif dl == 0:
        license_str = "do not have a driving licence"
    else:
        license_str = "have an unknown driving-licence status"

    ncars = int(row['NumberOfCars']) if pd.notna(row['NumberOfCars']) else 0
    if ncars == 0:
        car_str = "no car in the household"
    elif ncars == 1:
        car_str = "one car in the household"
    else:
        car_str = f"{ncars} cars in the household"

    pass_str = ("hold a public-transport season pass" if row['pt_pass'] == 1
                else "do not hold a public-transport season pass")
    bike_str = BICYCLE_MAP.get(int(row['Bicycle']), 'have unknown bicycle access')
    ptfreq_str = PTFREQ_MAP.get(int(row['PublicTransportFrequency']),
                                'have unknown public-transport usage')
    day_str = DATETYPE_MAP.get(str(row['DateType']), 'a typical day')

    occ_str  = OCCUPATION_MAP.get(int(row['Occupation']), 'of unknown occupation') \
               if pd.notna(row['Occupation']) else 'of unknown occupation'
    edu_str  = EDUCATION_MAP.get(int(row['Education']), 'an unknown level of education') \
               if pd.notna(row['Education']) else 'an unknown level of education'
    house_str = HOUSING_MAP.get(int(row['HousingType']), 'unknown housing')
    caravail_str = CARAVAIL_MAP.get(int(row['CarAvailabilityFrequency']), 'car availability is unknown') \
                   if pd.notna(row['CarAvailabilityFrequency']) else 'car availability is unknown'

    hh   = int(row['Household_Total'])  if pd.notna(row['Household_Total'])  else None
    kids = int(row['Household_0_6yrs']) if pd.notna(row['Household_0_6yrs']) else 0
    zone = row['TicketZone'] if pd.notna(row['TicketZone']) else None

    # 出行目的（出发地 → 目的地 地点类型）、出发时刻、距离
    origin = PLACE_MAP.get(int(row['Origin']), 'somewhere') if pd.notna(row.get('Origin')) else None
    dest   = PLACE_MAP.get(int(row['Destination']), 'somewhere') if pd.notna(row.get('Destination')) else None
    purpose_str = f"from {origin} to {dest}" if origin and dest else "a trip"
    dh = row.get('DepartureHour')
    dep_str = f"{int(dh):02d}:{int(round((dh - int(dh)) * 60)):02d}" if pd.notna(dh) else None
    dist = row.get('TripDistanceKm')
    dist_str = f"{dist:.1f} km" if pd.notna(dist) else None

    return dict(gender=gender, license_str=license_str, car_str=car_str,
                pass_str=pass_str, bike_str=bike_str, ptfreq_str=ptfreq_str,
                day_str=day_str, occ_str=occ_str, edu_str=edu_str, house_str=house_str,
                caravail_str=caravail_str, hh=hh, kids=kids, zone=zone,
                purpose_str=purpose_str, dep_str=dep_str, dist_str=dist_str)

def options_and_avail(row):
    lines, names = [], []
    for m in MODES:
        if int(row[f'av_{m}']) != 1:
            continue
        name = MODE_INT_TO_STR[m]
        lines.append(f"{name}: {row[f'tt_{m}']:.0f} min, €{row[f'cost_{m}']:.2f}")
        names.append(name)
    return lines, names

def trip_to_text(row, include_label: bool = True) -> str:
    t = get_traveller_info(row)
    opts, _ = options_and_avail(row)
    zone_txt = f", ticket zone {t['zone']}" if t['zone'] else ""
    when = f" departing {t['dep_str']}" if t['dep_str'] else ""
    dist = f", {t['dist_str']}" if t['dist_str'] else ""
    text = (
        f"Profile: age {int(row['Age'])}, {t['gender']}, {t['occ_str']}, {t['license_str']}, "
        f"{t['car_str']}.\n"
        f"Trip: {t['purpose_str']}{dist}, on {t['day_str']}{when}{zone_txt}.\n"
        f"Options — " + " | ".join(opts)
    )
    if include_label:
        text += f"\nCHOICE: {MODE_INT_TO_STR[int(row[LABEL_COL])]}"
    return text

def build_persona_system_msg(query_row, n_examples: int) -> str:
    t   = get_traveller_info(query_row)
    age = int(query_row['Age'])
    if age < 18:
        life_stage = "a young person"
    elif age < 25:
        life_stage = "a young adult"
    elif age < 40:
        life_stage = "in your prime working years"
    elif age < 55:
        life_stage = "middle-aged"
    elif age < 65:
        life_stage = "approaching retirement"
    else:
        life_stage = "a senior"

    hh_txt = ""
    if t['hh']:
        ppl = "person" if t['hh'] == 1 else "people"
        hh_txt = f" Your household has {t['hh']} {ppl}"
        hh_txt += f" including {t['kids']} young children." if t['kids'] else "."

    trip_bits = [t['purpose_str']]
    if t['dist_str']: trip_bits.append(f"about {t['dist_str']}")
    if t['dep_str']:  trip_bits.append(f"departing around {t['dep_str']}")
    trip_desc = ", ".join(trip_bits)

    persona = (
        f"You are a resident of the Helsinki region, {age} years old, {t['gender']}, "
        f"{life_stage}, {t['occ_str']}, with {t['edu_str']}. "
        f"You live in {t['house_str']}. You {t['license_str']} and have {t['car_str']} "
        f"({t['caravail_str']}). You {t['pass_str']}, you {t['bike_str']}, and you "
        f"{t['ptfreq_str']}.{hh_txt} "
        f"It is {t['day_str']}, and you are about to make a trip {trip_desc}. "
        f"Your available options for this trip are listed below."
    )
    system_msg = (
        "You are simulating a real traveller in the Helsinki region (HSL) making an "
        "everyday travel decision. Fully adopt the identity described below — think, "
        "feel, and reason as this specific person would. Consider your age, gender, "
        "occupation, household situation, whether it is a weekday or weekend, the trip "
        "purpose and distance, and what each available mode would cost you in time and money.\n\n"
        f"[YOUR IDENTITY]\n{persona}\n\n"
        f"Before deciding, you will see {n_examples} example trips made by other "
        "Helsinki-region residents in situations similar to yours. Learn from their "
        "choices, but make your own decision based on who YOU are and what makes sense "
        "for YOUR specific situation. Stay fully in character throughout — "
        "use 'I' when reasoning, not 'the traveller'."
    )
    return system_msg

def build_fewshot_prompt(query_row, examples_df: pd.DataFrame) -> list:
    n_examples = len(examples_df)
    messages = [{"role": "system",
                 "content": build_persona_system_msg(query_row, n_examples)}]

    for _, ex_row in examples_df.iterrows():
        messages.append({
            "role":    "user",
            "content": "Here is an example trip made by another Helsinki-region resident:\n\n"
                       + trip_to_text(ex_row, include_label=False),
        })
        messages.append({
            "role":    "assistant",
            "content": f"CHOICE: {MODE_INT_TO_STR[int(ex_row[LABEL_COL])]}",
        })

    t = get_traveller_info(query_row)
    opt_lines, avail_names = options_and_avail(query_row)
    options_block = "\n".join(f"- {line}" for line in opt_lines)
    allowed = ", ".join(avail_names)   # 只允许从可得模式中选
    zone_txt = f", ticket zone {t['zone']}" if t['zone'] else ""
    dist_txt = f" of about {t['dist_str']}" if t['dist_str'] else ""
    when_txt = f" departing around {t['dep_str']}" if t['dep_str'] else ""

    query_text = (
        f"Now it is your turn to decide. Remember, you ARE this person:\n\n"
        f"- I am {int(query_row['Age'])} years old, {t['gender']}, {t['occ_str']}\n"
        f"- I {t['license_str']}\n"
        f"- I have {t['car_str']} ({t['caravail_str']})\n"
        f"- I {t['pass_str']}\n"
        f"- My trip: {t['purpose_str']}, on {t['day_str']}{when_txt}{zone_txt}\n\n"
        f"My available options for this trip{dist_txt} (time in minutes, cost in euros):\n"
        f"{options_block}\n\n"
        f"As this person, decide what YOU would actually choose. Think about your age, "
        f"your physical comfort, your time, your wallet, whether you own a car or a pass, "
        f"the trip purpose and distance, and convenience.\n\n"
        f"First give a brief reason (in first person, e.g. \"I would choose ... because I ...\"), "
        f"then reply with ONLY one of these exact words: {allowed}.\n"
        f"Use this exact format: "
        f"\"REASON: [your reason here]. CHOICE: [{allowed}]\".\n"
    )
    messages.append({"role": "user", "content": query_text})
    return messages
