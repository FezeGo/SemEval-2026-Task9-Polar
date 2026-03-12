LANGUAGE_MAP = {
    'amh': 'Amharic',
    'arb': 'Arabic',
    'ben': 'Bengali',
    'deu': 'German',
    'eng': 'English',
    'fas': 'Persian',
    'hau': 'Hausa',
    'hin': 'Hindi',
    'ita': 'Italian',
    'khm': 'Khmer',
    'mya': 'Burmese',
    'nep': 'Nepali',
    'ori': 'Odia',
    'pan': 'Punjabi',
    'pol': 'Polish',
    'rus': 'Russian',
    'spa': 'Spanish',
    'swa': 'Swahili',
    'tel': 'Telugu',
    'tur': 'Turkish',
    'urd': 'Urdu',
    'zho': 'Chinese'
}

LABEL_COLUMNS = [
    "political",
    "racial/ethnic",
    "religious",
    "gender/sexual",
    "other"
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_COLUMNS)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_COLUMNS)}

NUM_LABELS = len(LABEL_COLUMNS)