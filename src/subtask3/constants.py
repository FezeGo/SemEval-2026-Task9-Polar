LANGUAGE_MAP = {
    'amh': 'Amharic',
    'arb': 'Arabic',
    'ben': 'Bengali',
    'deu': 'German',
    'eng': 'English',
    'fas': 'Persian',
    'hau': 'Hausa',
    'hin': 'Hindi',
    'khm': 'Khmer',
    'nep': 'Nepali',
    'ori': 'Odia',
    'pan': 'Punjabi',
    'spa': 'Spanish',
    'swa': 'Swahili',
    'tel': 'Telugu',
    'tur': 'Turkish',
    'urd': 'Urdu',
    'zho': 'Chinese'
}

LABEL_COLUMNS = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation",
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_COLUMNS)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_COLUMNS)}

NUM_LABELS = len(LABEL_COLUMNS)