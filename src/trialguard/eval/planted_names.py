"""Phase 40: cohort notes with one planted, unlabelled patient or relative name.

Measures the NER refusal's recall. Names are drawn from many origins and include
ones that are also English words, since those are where a name detector fails.
"""

from __future__ import annotations

import random
import re

FIRST = [
    "Amara", "Chidi", "Ngozi", "Kwame", "Fatima", "Omar", "Layla", "Yusuf", "Aisha",
    "Mei", "Wei", "Hiroshi", "Yuki", "Min-jun", "Seo-yeon", "Thanh", "Linh", "Arjun",
    "Priya", "Rohan", "Ananya", "Sanjay", "Deepa", "Maria", "Jose", "Lucia", "Diego",
    "Camila", "Mateo", "Sofia", "John", "Mary", "James", "Patricia", "Robert", "Linda",
    "Michael", "Barbara", "David", "Susan", "Olga", "Dmitri", "Svetlana", "Ivan",
    "Ingrid", "Lars", "Sione", "Mere", "Tane", "Aroha", "Jarrah", "Kirra", "Hope",
    "Grace", "Faith", "Joy", "Rose", "Hunter", "Sky", "Bill", "Will", "Dawn",
]
LAST = [
    "Okafor", "Adeyemi", "Mensah", "Haddad", "Khan", "Rahman", "Nguyen", "Tran",
    "Chen", "Wang", "Li", "Zhang", "Tanaka", "Sato", "Kim", "Park", "Lee", "Patel",
    "Sharma", "Iyer", "Reddy", "Singh", "Gonzalez", "Rodriguez", "Hernandez", "Lopez",
    "Silva", "Costa", "Rossi", "Bianchi", "Smith", "Johnson", "Williams", "Brown",
    "Jones", "Miller", "Davis", "Wilson", "Taylor", "Thompson", "Ivanova", "Petrov",
    "Kowalski", "Novak", "Larsen", "Andersson", "Tupou", "Ngata", "Walker", "Young",
    "Long", "King", "Hill", "Wood", "Stone", "Moon", "Bird", "Fox", "Bell", "Rice",
    "Cross", "Brooks",
]
TEMPLATES = 6
# Written and frozen before the detector's rules, never used to tune them:
# recall on these is the number that says whether the rules generalise.
HOLDOUT = (
    "{note} Accompanied today by {first} {last}.",
    "{first} {last}'s presenting complaint: {note}",
    "{note} Spoke with {first} by phone about the results.",
    "{note} Next of kin: {first} {last}.",
    "{note} {first} {last} prefers morning appointments.",
    "Seen with {first} {last} (partner). {note}",
)
# Frozen after HOLDOUT showed the first rules overfit (recall 0.76), before any
# change made in response. HOLDOUT has now informed the detector; this has not.
HOLDOUT_2 = (
    "{note} Discussed the plan with {first} at the bedside.",
    "{note} {first} asked about trial travel costs.",
    "Called {first} {last} with the biopsy result. {note}",
    "{note} Transport arranged by {first}.",
    "{note} Consent witnessed by {first} {last}.",
    "{note} Message left for {first} {last}.",
)
EXCLUDED_BASES = ("sigir-201515", "trec-202148")  # these notes already name the patient
_LEAD = re.compile(r"^(?:An?|The)\s+(?=\d)", re.I)


def plant(note: str, first: str, last: str, template: int) -> tuple[str, str]:
    """The note with one name planted, and the name as it appears."""
    if template == 0:
        name = f"{first} {last}"
        body = _LEAD.sub("", note, count=1)
        return f"{name} is a {body}" if body != note else f"{name}. {note}", name
    if template == 1:
        name = f"{last}"
        return f"Mr. {name} was seen today. {note}", name
    if template == 2:
        name = f"{first}"
        return f"{note} Her daughter {name} provides most of the history.", name
    if template == 3:
        name = f"{first} {last}"
        return f"{note} Lives with his wife, {name}, who drives him to clinic.", name
    if template == 4:
        name = f"{last}"
        return f"{note} Ms. {name} declined further imaging.", name
    name = f"{first} {last}"
    return f"{note} Referred by her sister {name} after the diagnosis.", name


def plant_holdout(
    note: str, first: str, last: str, template: int, forms: tuple[str, ...] = HOLDOUT
) -> tuple[str, str]:
    form = forms[template]
    name = f"{first} {last}" if "{last}" in form else first
    return form.format(note=note, first=first, last=last), name


def build(seed: int = 40, per_template: int = 40, holdout: int = 0) -> list[dict]:
    """holdout=0 is the tuning set, 1 and 2 the frozen held-out template sets."""
    from trialguard.eval.cohorts import load_patients

    rng = random.Random(seed)  # noqa: S311 -- reproducible sampling, not security
    bases = [
        (p["patient_id"], p["description"])
        for c in ("sigir", "trec_2021", "trec_2022", "trec_2023")
        for p in load_patients(c)
        if p["patient_id"] not in EXCLUDED_BASES
    ]
    out = []
    if holdout:
        rng = random.Random(seed + holdout)  # noqa: S311
    forms = {1: HOLDOUT, 2: HOLDOUT_2}.get(holdout)
    for template in range(len(forms) if forms else TEMPLATES):
        for base_id, note in rng.sample(bases, per_template):
            first, last = rng.choice(FIRST), rng.choice(LAST)
            if forms:
                text, name = plant_holdout(note, first, last, template, forms)
            else:
                text, name = plant(note, first, last, template)
            out.append({"base_id": base_id, "template": template, "name": name, "note": text})
    return out
