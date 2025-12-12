"""
Constants for import in ./evaluator.py
"""

DIALECTS = [
    "Algeria",
    "Bahrain",
    "Egypt",
    "Iraq",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Libya",
    "Morocco",
    "Oman",
    "Palestine",
    "Qatar",
    "Saudi_Arabia",
    "Sudan",
    "Syria",
    "Tunisia",
    "UAE",
    "Yemen",
]
assert len(DIALECTS) == 18

COUNTRY2DIALECT = {
    "dza": "Algeria",
    "bhr": "Bahrain",
    "egy": "Egypt",
    "irq": "Iraq",
    "jor": "Jordan",
    "kwt": "Kuwait",
    "lbn": "Lebanon",
    "lby": "Libya",
    "mar": "Morocco",
    "omn": "Oman",
    "pse": "Palestine",
    "qat": "Qatar",
    "sau": "Saudi_Arabia",
    "sdn": "Sudan",
    "syr": "Syria",
    "tun": "Tunisia",
    "are": "UAE",
    "yem": "Yemen",
}
DIALECT2COUNTRY = {COUNTRY2DIALECT[key]: key for key in COUNTRY2DIALECT}

COUNTRY2MACRO_DIALECT = {
    "dza": "Maghreb",
    "bhr": "Gulf",
    "egy": "Nile",
    "irq": "Gulf",
    "jor": "Levant",
    "kwt": "Gulf",
    "lbn": "Levant",
    "lby": "Maghreb",
    "mar": "Maghreb",
    "omn": "Gulf",
    "pse": "Levant",
    "qat": "Gulf",
    "sau": "Gulf",
    "sdn": "Nile",
    "syr": "Levant",
    "tun": "Maghreb",
    "are": "Gulf",
    "yem": "Aden",
}

DIALECTS8 = ["dza", "mar", "egy", "sdn", "pse", "syr", "sau", "kwt"]
DIALECTS4 = ["dza", "egy", "syr", "sau"]
MT_DIRECTIONS = list(map(lambda x: f"eng-{x}", DIALECTS4)) + list(
        map(lambda x: f"msa-{x}", DIALECTS4)
    ) + list(
        map(lambda x: f"{x}-eng", DIALECTS4)
    ) + list(
        map(lambda x: f"{x}-msa", DIALECTS4)
    )
TASK2DIALECTS = {
    "monolingual": DIALECTS8,
    "crosslingual": DIALECTS8, 
    "mt": MT_DIRECTIONS
}

MICROLANGUAGE_MAP = {
    "ara": [
        "ara", 
        "arb", 
        "arq", 
        "arz",
        "acm", 
        "ayp", 
        "apc", 
        "ajp", 
        "afb", 
        "ayl", 
        "ary", 
        "acw", 
        "ars", 
        "apd", 
        "aeb",
        "ayn", 
        "acq", # Could be added to 
    ],
    "eng": [
        "eng_Latn"
    ]
}
