# scripts/generate_synthetic.py
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Settings
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REAL_INPUT = OUT_DIR / "events_real.csv"   # optional: if you have a separate real file
EXISTING_INPUT = OUT_DIR / "events.csv"    # will be read if present (kept as "real")
OUT_MERGED = OUT_DIR / "events.csv"
OUT_SYN = OUT_DIR / "synthetic_events.csv"

N_SYN = 200
RANDOM = True  # random each run (you wanted random)
INCLUDE_MEGA = True
NUM_MEGA = 2

# Event-type attendance caps (min,max)
TYPE_CAPS = {
    "Practice": (10, 30),
    "GBM": (30, 100),
    "Special": (20, 300),
    "Collab": (30, 140),
    "Fundraiser": (10, 80),
    "Trip": (20, 80),
    "Rehearsal": (40, 60),  # your request: rehearsals/hellweek around 40-60
}

# Base probabilities for event types (approx)
TYPE_WEIGHTS = [
    ("GBM", 0.30),
    ("Special", 0.18),
    ("Collab", 0.15),
    ("Practice", 0.18),
    ("Fundraiser", 0.10),
    ("Trip", 0.04),
    ("Rehearsal", 0.05),
]

# Locations pool (popular ones from your data)
LOCATIONS = [
    "ILSB 233", "ILSB 116A", "UC Ballroom", "RAC/UC Ballroom",
    "Quad/Commons", "Practice Room", "External Venue",
    "Theatre", "The Commons", "M&P 104"
]

# Calendar pattern: months with heavier event density
# We'll use your semester: Sep-Nov heavy, Feb-May heavy, Jan/Dec light
MONTH_WEIGHTS = {
    1: 0.02, 2: 0.10, 3: 0.12, 4: 0.12, 5: 0.08, 6: 0.01,
    7: 0.01, 8: 0.03, 9: 0.18, 10: 0.15, 11: 0.12, 12: 0.06
}

# Helper distributions for likes/comments
def sample_likes(event_type, is_major=False):
    # base ranges by type
    if is_major:
        return int(max(100, np.random.lognormal(mean=5.5, sigma=0.6)))  # large likes for mega
    base = {
        "Practice": (5, 60),
        "GBM": (20, 200),
        "Special": (40, 700),
        "Collab": (30, 400),
        "Fundraiser": (5, 150),
        "Trip": (10, 120),
        "Rehearsal": (20,80),
    }.get(event_type, (10,150))
    low, high = base
    # skew with lognormal
    val = int(np.clip(np.random.lognormal(mean=np.log((low+high)/2+1), sigma=0.9), low, high*3))
    return val

def sample_comments(likes):
    # comments correlate with likes but weaker
    # baseline comments ~ likes * factor (0.02 - 0.12) plus noise
    factor = random.uniform(0.02, 0.12)
    return int(max(0, np.random.poisson(likes * factor)))

def clamp_attendance(x, low, high):
    return int(max(low, min(high, round(x))))

# Non-linear attendance function: uses likes, comments, type baseline, food, collab, weekend, competition
def generate_attendance(event_type, likes, comments, has_food, is_collab, is_weekend, is_major, same_week_count, competition_intensity):
    # base attendance from type (midpoint)
    cap_low, cap_high = TYPE_CAPS.get(event_type, (20, 120))
    base = (cap_low + cap_high) / 2.0

    # likes effect (non-linear): sqrt + diminishing returns
    likes_effect = (likes ** 0.5) * 1.5

    # comments effect (stronger signal per item)
    comments_effect = comments * 2.2

    # food and collab boosts
    food_boost = 20 if has_food else 0
    collab_boost = 25 if is_collab else 0
    major_boost = 0
    if is_major:
        # big multiplier for major events
        major_boost = cap_high * 0.7

    weekend_boost = 12 if is_weekend else 0

    # penalty for many same-week events (fatigue)
    fatigue = max(0, same_week_count - 1) * -8

    # competition intensity (other orgs that week) reduces turnout
    competition_penalty = -5 * competition_intensity

    # small random noise
    noise = np.random.normal(0, cap_high * 0.08)

    raw = base + likes_effect + comments_effect + food_boost + collab_boost + major_boost + weekend_boost + fatigue + competition_penalty + noise

    # clamp to type caps but allow high if is_major or mega
    if is_major:
        max_allowed = max(cap_high * 3, 400)
    else:
        max_allowed = cap_high
    min_allowed = cap_low
    return clamp_attendance(raw, min_allowed, int(max_allowed))

# Calendar helpers: pick dates following your pattern
def weighted_month_choice():
    months = list(MONTH_WEIGHTS.keys())
    weights = np.array([MONTH_WEIGHTS[m] for m in months], dtype=float)
    weights = weights / weights.sum()
    return random.choices(months, weights=weights, k=1)[0]

def random_date_in_month(month, year=2025):
    # generate day with bias towards mid-semester weeks (10-25)
    day = random.randint(1, 28)
    return datetime(year, month, day)

# Read existing real events (if exist), prefer events_real.csv, else events.csv if present
real_df = None
if REAL_INPUT.exists():
    real_df = pd.read_csv(REAL_INPUT)
elif EXISTING_INPUT.exists():
    # if events.csv exists already and seems to be the original small sample, treat as real input
    real_df = pd.read_csv(EXISTING_INPUT)
else:
    # create a minimal skeleton from your example if nothing found
    real_df = pd.DataFrame([
        # keep lightweight if no real data present - user already has real rows elsewhere
    ])

# build synthetic rows
syn_rows = []
for i in range(N_SYN):
    # choose event type
    types, w = zip(*TYPE_WEIGHTS)
    event_type = random.choices(types, weights=w, k=1)[0]

    # determine if major or mega
    is_major = False
    if event_type == "Special" and random.random() < 0.08:  # ~8% special events are "major"
        is_major = True

    # include specific mega events occasionally (NUM_MEGA total)
    is_mega = False
    # we'll add NUM_MEGA mega events after the loop populating some at the end

    month = weighted_month_choice()
    date = random_date_in_month(month)
    dow = date.weekday()
    is_weekend = dow >= 5

    # same-week counts & competition intensity (simulate from calendar density)
    same_week_count = random.choices([1,2,3], weights=[0.7,0.22,0.08], k=1)[0]
    competition_intensity = random.choices([0,1,2,3], weights=[0.5,0.28,0.15,0.07], k=1)[0]

    # has food with higher chance for Special & Fundraiser
    has_food = random.random() < (0.55 if event_type in ["Special","Fundraiser","Collab"] else 0.18)

    # collab chance
    is_collab = random.random() < (0.35 if event_type == "Collab" else 0.12)

    # likes/comments
    likes = sample_likes(event_type, is_major)
    comments = sample_comments(likes)

    # derive attendance
    attendance = generate_attendance(event_type, likes, comments, has_food, is_collab, is_weekend, is_major, same_week_count, competition_intensity)

    name = f"{event_type} Event {i+1}"
    loc = random.choice(LOCATIONS)
    notes = ""
    syn_rows.append({
        "date": date.strftime("%Y-%m-%d"),
        "event_name": name,
        "event_type": event_type,
        "attendance": attendance,
        "likes": likes,
        "comments": comments,
        "location": loc,
        "notes": notes,
        "source": "synthetic"
    })

# Add a couple of mega events (very large attendance) if desired
if INCLUDE_MEGA:
    for m in range(NUM_MEGA):
        month = random.choice([4,5,9])  # popular months for big shows
        date = random_date_in_month(month)
        likes = int(np.random.lognormal(mean=7.0, sigma=0.4))  # big likes
        comments = max(10, sample_comments(likes))
        attendance = int(np.clip(likes * random.uniform(0.2, 0.6), 200, 700))
        syn_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "event_name": f"MEGA Cultural Event {m+1}",
            "event_type": "Special",
            "attendance": attendance,
            "likes": likes,
            "comments": comments,
            "location": random.choice(LOCATIONS),
            "notes": "Mega event (cultural showcase)",
            "source": "synthetic"
        })

# Build dataframes and merge with real_df
syn_df = pd.DataFrame(syn_rows)

# Normalize column names and cast types
for col in ["attendance","likes","comments"]:
    if col in syn_df.columns:
        syn_df[col] = syn_df[col].astype(int)

# If real_df has rows, preserve them and label source
if real_df is not None and not real_df.empty:
    real_df = real_df.copy()
    # ensure expected columns exist in real_df
    for col in ["date","event_name","event_type","attendance","likes","comments","location","notes"]:
        if col not in real_df.columns:
            real_df[col] = ""
    real_df["source"] = real_df.get("source", "real")
    merged = pd.concat([real_df, syn_df], ignore_index=True, sort=False)
else:
    merged = syn_df.copy()

# Shuffle the merged set to avoid ordering giveaways
merged = merged.sample(frac=1, random_state=None).reset_index(drop=True)

# Save outputs
merged.to_csv(OUT_MERGED, index=False)
syn_df.to_csv(OUT_SYN, index=False)
merged.to_csv(OUT_DIR / "events_source_merged.csv", index=False)

print(f"Generated {len(syn_df)} synthetic events and saved merged dataset to {OUT_MERGED}")
print(f"Synthetic-only saved to {OUT_SYN}")
