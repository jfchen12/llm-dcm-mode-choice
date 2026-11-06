"""Create illustrative records from hand-written rules, without reading real data."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def make_dataset(dataset, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for person in range(100):
        age = int(rng.integers(18, 81))
        female = int(rng.integers(0, 2))
        licence = int(rng.random() < .8)
        cars = int(rng.integers(0, 3)) if licence else 0
        season_pass = int(rng.random() < .45)
        bicycle = int(rng.choice([1, 2, 3, 4]))
        occupation = 4 if age >= 67 else (3 if age < 24 else int(rng.choice([1, 1, 1, 2])))
        education = int(rng.integers(2, 7))
        housing = int(rng.integers(1, 4))
        household = int(rng.integers(1, 5))
        children = int(rng.integers(0, household)) if 25 <= age <= 60 else 0
        pt_frequency = 1 if season_pass else int(rng.integers(2, 6))
        destination = 2 if occupation == 1 else (3 if occupation == 3 else 5)
        for trip in range(2):
            km = round(float(np.clip(rng.lognormal(1.2, .8), .4, 28)), 2)
            hour = int(rng.integers(6, 23))
            walk = km / 4.5
            cycle = km / 14
            drive = km / float(rng.uniform(20, 40)) + .07
            access = float(rng.uniform(.04, .12))
            rail_fraction = float(rng.uniform(.2, .8))
            pt_ride = km / 24
            interchange = .06 if km > 6 else 0.
            pt = access + pt_ride + interchange
            cost_pt = 0. if season_pass and dataset == 'hrts' else 2.8 + .08 * km
            cost_drive = .2 * km
            row = {'trip_id': f'syn_{dataset}_{person:03d}_{trip}',
                   'traveller_id': f'syn_{dataset}_person_{person:03d}', 'is_synthetic': True}
            if dataset == 'hrts':
                row.update(Age=age, Gender=1 if female else 2, HasDrivingLicense=licence,
                           NumberOfCars=cars, pt_pass=season_pass, Bicycle=bicycle,
                           PublicTransportFrequency=pt_frequency, DateType='ARKI' if person % 4 else 'VKOLOPPU',
                           Occupation=occupation, Education=education, HousingType=housing,
                           CarAvailabilityFrequency=1 if cars else 4, Household_Total=household,
                           Household_0_6yrs=children, TicketZone='AB' if km < 10 else 'ABC',
                           Origin=1 if trip == 0 else destination, Destination=destination if trip == 0 else 1,
                           DepartureHour=hour, TripDistanceKm=km)
                for code, time, cost, available in [(1, drive, cost_drive, int(licence and cars > 0)),
                                                  (2, pt, cost_pt, 1), (3, cycle, 0, int(bicycle != 4)),
                                                  (4, walk, 0, 1)]:
                    row.update({f'tt_{code}': round(time * 60, 3), f'cost_{code}': round(cost, 2),
                                f'av_{code}': available})
                times = np.array([drive, pt, cycle, walk])
                costs = np.array([cost_drive, cost_pt, 0., 0.])
                constants = np.array([.2, .2 + .5 * season_pass, -.1, .8])
                available = np.array([row[f'av_{i}'] for i in range(1, 5)], dtype=bool)
                label = 'choice_mode'
            else:
                charge = 4. if person % 9 == 0 else 0.
                row.update(age=age, female=female, driving_license=licence, car_ownership=cars,
                           purpose=int(rng.choice([1, 3, 4, 5] if occupation == 1 else
                                                  [2, 3, 5] if occupation == 3 else [3, 5])),
                           start_time=hour, distance=round(km * 1000),
                           dur_walking=round(walk, 5), dur_cycling=round(cycle, 5),
                           dur_pt_access=round(access, 5), dur_pt_rail=round(pt_ride * rail_fraction, 5),
                           dur_pt_bus=round(pt_ride * (1 - rail_fraction), 5), dur_pt_int=interchange,
                           pt_interchanges=int(interchange > 0), dur_driving=round(drive, 5),
                           cost_transit=round(cost_pt, 2), cost_driving_fuel=round(cost_drive, 2),
                           cost_driving_ccharge=charge, driving_traffic_percent=round(float(rng.uniform(0, .7)), 3))
                times = np.array([walk, cycle, pt, drive])
                costs = np.array([0., 0., cost_pt, cost_drive + charge])
                constants = np.array([.8, -.1, .3, .2 if licence and cars else -2.])
                available = np.ones(4, dtype=bool)  # Matches the illustrative LPMC choice set.
                label = 'travel_mode'
            utility = constants - 2.0 * times - .15 * costs + rng.gumbel(size=4)
            utility[~available] = -np.inf
            row[label] = int(utility.argmax() + 1)
            rows.append(row)
    frame = pd.DataFrame(rows)
    train, evaluation = train_test_split(np.arange(len(frame)), test_size=.1,
                                         stratify=frame[label], random_state=seed)
    frame['sample_role'] = 'train'
    frame.loc[evaluation, 'sample_role'] = 'evaluation'
    assert len(train) == 180 and len(evaluation) == 20
    return frame


def write_examples(frame, dataset, destination):
    label = 'choice_mode' if dataset == 'hrts' else 'travel_mode'
    evaluation = frame[frame.sample_role == 'evaluation'].copy()
    evaluation[['trip_id', label]].to_csv(destination / f'{dataset}_evaluation_labels.csv', index=False)
    evaluation[['trip_id']].assign(predicted_mode='').to_csv(destination / f'{dataset}_prediction_template.csv', index=False)
    predictions = []
    for _, row in evaluation.iterrows():
        if dataset == 'hrts':
            times = [row[f'tt_{i}'] if row[f'av_{i}'] else np.inf for i in range(1, 5)]
        else:
            times = [row.dur_walking, row.dur_cycling,
                     row.dur_pt_access + row.dur_pt_rail + row.dur_pt_bus + row.dur_pt_int, row.dur_driving]
        predictions.append(int(np.argmin(times) + 1))
    # Deliberately simple placeholder: no LLM call and no target-label access.
    evaluation[['trip_id']].assign(predicted_mode=predictions, prediction_source='mock_fastest_available_mode').to_csv(
        destination / f'{dataset}_mock_predictions.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/synthetic')
    parser.add_argument('--examples-dir', type=Path, default=ROOT / 'examples')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.examples_dir.mkdir(parents=True, exist_ok=True)
    for dataset, seed in [('hrts', 20261103), ('lpmc', 20261104)]:
        data = make_dataset(dataset, seed)
        data.to_csv(args.output_dir / f'{dataset}_synthetic.csv', index=False)
        write_examples(data, dataset, args.examples_dir)
        print(f'{dataset}: {len(data)} synthetic rows; 180 train / 20 evaluation')
