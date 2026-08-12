"""Ten kinds of fact, ten values each. A hundred items.

Six items was a pilot. A probe trained on six examples separates anything, and
the direction it finds says more about the fit than about the model.

Every fact is one clause in the same frame, so the `absent` condition can be
built by putting a *different* kind of fact in the same slot. That matters: if
`absent` is built by deleting the sentence, the two classes differ in their text
and a probe reads the tokens rather than the state. The pilot did exactly that
and scored 6/6 at layer 0, which is the embedding layer and cannot hold a state.
"""

from __future__ import annotations

TYPES = [
    ("dog", "my dog is called {v}", "What is my dog called?",
     ["Bagr", "Fizzle", "Kudla", "Wobbet", "Trixel", "Marnok", "Bristle", "Yorrick", "Plunk", "Skarn"]),
    ("order", "my order number is {v}", "What is my order number?",
     ["4417", "8290", "1053", "7734", "6621", "3908", "5142", "2867", "9375", "1486"]),
    ("city", "I live in {v}", "Which city do I live in?",
     ["Brno", "Utrecht", "Bergen", "Kaunas", "Porto", "Graz", "Tampere", "Rijeka", "Aarhus", "Leon"]),
    ("error", "the error code I got was {v}", "Which error code did I get?",
     ["E-88", "X-12", "F-401", "Q-73", "B-206", "R-95", "T-340", "K-17", "M-628", "Z-54"]),
    ("time", "my train leaves at {v}", "When does my train leave?",
     ["19:40", "06:15", "13:52", "21:08", "11:27", "04:36", "17:49", "08:03", "22:41", "15:14"]),
    ("account", "my account number ends in {v}", "What does my account number end in?",
     ["302", "718", "455", "961", "233", "807", "574", "129", "686", "340"]),
    ("allergy", "I am allergic to {v}", "What am I allergic to?",
     ["walnuts", "penicillin", "shellfish", "latex", "mustard", "sesame", "sulfites", "kiwi", "wasp stings", "aspirin"]),
    ("room", "my hotel room is {v}", "Which hotel room am I in?",
     ["614", "227", "1108", "355", "902", "471", "1236", "588", "743", "196"]),
    ("flight", "my flight is {v}", "What is my flight number?",
     ["QR318", "LH772", "BA945", "AY206", "OK483", "SK651", "TP129", "AZ874", "EW537", "IB260"]),
    ("cat", "my cat is called {v}", "What is my cat called?",
     ["Miso", "Grendel", "Pilch", "Nectarine", "Bobbin", "Vosk", "Tamarind", "Klepto", "Sorrel", "Nubbin"]),
]


def build_items():
    """100 items. `donor` is a clause of a different kind, for the matched
    `absent` condition — same frame, same shape, wrong fact."""
    out = []
    for ti, (key, clause, ask, values) in enumerate(TYPES):
        d_key, d_clause, _, d_values = TYPES[(ti + 1) % len(TYPES)]
        for vi, v in enumerate(values):
            out.append({
                "key": f"{key}:{v}",
                "type": key,
                "value": v,
                "told": f"By the way, {clause.format(v=v)}.",
                "ask": ask,
                "donor": f"By the way, {d_clause.format(v=d_values[vi])}.",
            })
    return out


ITEMS100 = build_items()
