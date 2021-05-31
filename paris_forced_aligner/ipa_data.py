from typing import Tuple, Dict

import torch 
from torch import Tensor

arpabet_to_ipa = {
"AA": "ɑ",
"AE": "æ",
"AH": "ʌ",
"AO": "ɔ",
"AW": "aʊ",
"AX": "ə",
"AXR": "ɚ",
"AY": "aɪ",
"EH": "ɛ",
"ER": "ɝ",
"EY": "eɪ",
"IH": "ɪ",
"IX": "ɨ",
"IY": "i",
"OW": "oʊ",
"OY": "ɔɪ",
"UH": "ʊ",
"UW": "u",
"UX": "ʉ",
"B": "b ",
"CH": "tʃ",
"D": "d ",
"DH": "ð",
"DX": "ɾ",
"EL": "l̩",
"EM": "m̩",
"EN": "n̩",
"F": "f",
"G": "ɡ",
"HH": "h",
"H": "h",
"JH": "dʒ",
"K": "k",
"L": "l",
"M": "m",
"N": "n",
"NG": "ŋ",
"NX": "ɾ̃",
"P": "p",
"Q": "ʔ",
"R": "ɹ",
"S": "s",
"SH": "ʃ",
"T": "t",
"TH": "θ",
"V": "v",
"W": "w",
"WH ": "ʍ",
"Y": "j",
"Z": "z",
"ZH": "ʒ"}

CONSONANT_PLACES = ["bilabial", "labio-dental", "dental", "alveolar", "post-alveolar", "retroflex", "palatal", "velar", "uvular", "epi-glottal", "glottal", "co-articulated"]
CONSONANT_MANNERS = ["nasal", "plosive", "sibilant", "fricative", "affricate", "approximant", "tap", "trill", "lateral fricative", "lateral approximant", "lateral tap"]
CONSONANT_SECOND = ["labialized", "pharyngealized", "palatalized"]
CONSONANT_NON_PULMONIC = ["ejective", "implosive", "click"]
CONSONANT_VOWEL = ["consonant", "vowel"]
VOICING = ["voiced", "unvoiced"]
VOWEL_HEIGHT = ["high", "near-high", "mid-high", "mid", "low-mid", "near-low", "low"]
VOWEL_BACKNESS = ["front", "central", "back"]
VOWEL_ROUND = ["round", "unround"]
LENGTH = ["short", "long"]


#idx=0 for blank, idx=1 for when the feature is not relevant
CONSONANT_PLACES_IDX = {x: i+2 for i, x in enumerate(CONSONANT_PLACES)}
CONSONANT_MANNERS_IDX = {x: i+2 for i, x in enumerate(CONSONANT_MANNERS)}
CONSONANT_SECOND_IDX = {x: i+2 for i, x in enumerate(CONSONANT_SECOND)}
CONSONANT_NON_PULMONIC_IDX = {x: i+2 for i, x in enumerate(CONSONANT_PLACES)}
CONSONANT_VOWEL_IDX = {x: i+1 for i, x in enumerate(CONSONANT_VOWEL)}
VOICING_IDX = {x: i+1 for i, x in enumerate(VOICING)}

VOWEL_HEIGHT_IDX = {x: i+2 for i, x in enumerate(VOWEL_HEIGHT)}
VOWEL_BACKNESS_IDX = {x: i+2 for i, x in enumerate(VOWEL_BACKNESS)}
VOWEL_ROUND_IDX = {x: i+2 for i, x in enumerate(VOWEL_ROUND)}

LENGTH_IDX = {x: i+1 for i, x in enumerate(LENGTH)}

VOCAB_SIZES = {"consonant_places": len(CONSONANT_PLACES)+2,
        "consonant_manners": len(CONSONANT_MANNERS)+2,
        "consonant_non_pulmonic": len(CONSONANT_NON_PULMONIC)+2,
        "consonant_second": len(CONSONANT_SECOND)+2,
        "consonant_vowel": len(CONSONANT_VOWEL)+1,
        "voicing": len(VOICING)+1,
        "vowel_height": len(VOWEL_HEIGHT) + 2,
        "vowel_backness": len(VOWEL_BACKNESS) + 2,
        "vowel_round": len(VOWEL_ROUND) + 2,
        "length": len(LENGTH)+1}
BLANKLESS_FEATURES = ["length", "voicing", "consonant_vowel"]
CHARACTER_MAPPINGS = {
        "m": ("bilabial", "nasal", None, "consonant", "voiced", None, None, None),
        "p": ("bilabial", "plosive", None, "consonant", "unvoiced", None, None, None),
        "b": ("bilabial", "plosive", None, "consonant", "voiced", None, None, None),
        "ɸ": ("bilabial", "fricative", None, "consonant", "unvoiced", None, None, None),
        "β": ("bilabial", "fricative", None, "consonant", "voiced", None, None, None),
        "ⱱ̟": ("bilabial", "tap", None, "consonant", "voiced", None, None, None),
        "ʙ̥": ("bilabial", "trill", None, "consonant", "unvoiced", None, None, None),
        "ʙ": ("bilabial", "trill", None, "consonant", "voiced", None, None, None),
        "ɱ": ("labio-dental", "nasal", None, "consonant", "voiced", None, None, None),
        "p̪": ("labio-dental", "plosive", None, "consonant", "unvoiced", None, None, None),
        "b̪": ("labio-dental", "plosive", None, "consonant", "voiced", None, None, None),
        "f": ("labio-dental", "fricative", None, "consonant", "unvoiced", None, None, None),
        "v": ("labio-dental", "fricative", None, "consonant", "voiced", None, None, None),
        "ʋ": ("labio-dental", "approximant", None, "consonant", "voiced", None, None, None),
        "ⱱ": ("labio-dental", "tap", None, "consonant", "voiced", None, None, None),
        "n̥": ("alveolar", "nasal", None, "consonant", "unvoiced", None, None, None),
        "n": ("alveolar", "nasal", None, "consonant", "voiced", None, None, None),
        "t": ("alveolar", "plosive", None, "consonant", "unvoiced", None, None, None),
        "d": ("alveolar", "plosive", None, "consonant", "voiced", None, None, None),
        "s": ("alveolar", "sibilant", None, "consonant", "unvoiced", None, None, None),
        "z": ("alveolar", "sibilant", None, "consonant", "voiced", None, None, None),
        "ʃ": ("post-alveolar", "sibilant", None, "consonant", "unvoiced", None, None, None),
        "ʒ": ("post-alveolar", "sibilant", None, "consonant", "voiced", None, None, None),
        "t̪": ("dental", "plosive", None, "consonant", "unvoiced", None, None, None),
        "d̪": ("dental", "plosive", None, "consonant", "voiced", None, None, None),
        "θ": ("dental", "fricative", None, "consonant", "unvoiced", None, None, None),
        "ð": ("dental", "fricative", None, "consonant", "voiced", None, None, None),
        "ɹ": ("alveolar", "approximant", None, "consonant", "voiced", None, None, None),
        "ɾ̥": ("alveolar", "tap", None, "consonant", "unvoiced", None, None, None),
        "ɾ": ("alveolar", "tap", None, "consonant", "voiced", None, None, None),
        "r̥": ("alveolar", "trill", None, "consonant", "unvoiced", None, None, None),
        "r": ("alveolar", "trill", None, "consonant", "voiced", None, None, None),
        "ɬ": ("alveolar", "lateral fricative", None, "consonant", "unvoiced", None, None, None),
        "ɮ": ("alveolar", "lateral fricative", None, "consonant", "voiced", None, None, None),   	
        "l": ("alveolar", "lateral approximant", None, "consonant", "unvoiced", None, None, None),
        "ɺ̥": ("alveolar", "lateral tap", None, "consonant", "unvoiced", None, None, None),
        "ɺ": ("alveolar", "lateral tap", None, "consonant", "voiced", None, None, None),
        "ɳ̊": ("retroflex", "nasal", None, "consonant", "unvoiced", None, None, None),
        "ɳ": ("retroflex", "nasal", None, "consonant", "voiced", None, None, None),
        "ʈ": ("retroflex", "plosive", None, "consonant", "unvoiced", None, None, None),
        "ɖ": ("retroflex", "plosive", None, "consonant", "voiced", None, None, None),
        "ʂ": ("retroflex", "sibilant", None, "consonant", "unvoiced", None, None, None),
        "ʐ": ("retroflex", "sibilant", None, "consonant", "voiced", None, None, None),
        "ɻ": ("retroflex", "approximant", None, "consonant", "voiced", None, None, None),
        "ɭ": ("retroflex", "lateral approximant", None, "consonant", "voiced", None, None, None),
        "ɭ̥̆": ("retroflex", "lateral tap", None, "consonant", "unvoiced", None, None, None),
        "ɭ̆": ("retroflex", "lateral tap", None, "consonant", "voiced", None, None, None),
        "ɲ̊": ("palatal", "nasal", None, "consonant", "unvoiced", None, None, None),
        "ɲ": ("palatal", "nasal", None, "consonant", "voiced", None, None, None),
        "c": ("palatal", "plosive", None, "consonant", "unvoiced", None, None, None),
        "ɟ": ("palatal", "plosive", None, "consonant", "voiced", None, None, None),
        "ɕ": ("palatal", "sibilant", None, "consonant", "unvoiced", None, None, None),
        "ʑ": ("palatal", "sibilant", None, "consonant", "voiced", None, None, None),
        "ç": ("palatal", "fricative", None, "consonant", "unvoiced", None, None, None),
        "ʝ": ("palatal", "fricative", None, "consonant", "voiced", None, None, None),
        "j": ("palatal", "approximant", None, "consonant", "voiced", None, None, None),
        "ʎ": ("palatal", "lateral approximant", None, "consonant", "voiced", None, None, None),
        "ŋ̊": ("velar", "nasal", None, "consonant", "unvoiced", None, None, None),
        "ŋ": ("velar", "nasal", None, "consonant", "voiced", None, None, None), 	
        "k": ("velar", "plosive", None, "consonant", "unvoiced", None, None, None),
        "g": ("velar", "plosive", None, "consonant", "voiced", None, None, None),
        "x": ("velar", "fricative", None, "consonant", "unvoiced", None, None, None),
        "ɣ": ("velar", "fricative", None, "consonant", "voiced", None, None, None),
        "ɰ": ("velar", "approximant", None, "consonant", "voiced", None, None, None),
        "ɴ": ("uvular", "nasal", None, "consonant", "voiced", None, None, None), 	
        "q": ("uvular", "plosive", None, "consonant", "unvoiced", None, None, None),
        "ɢ": ("uvular", "plosive", None, "consonant", "voiced", None, None, None),
        "χ": ("uvular", "fricative", None, "consonant", "unvoiced", None, None, None),
        "ʁ": ("uvular", "fricative", None, "consonant", "voiced", None, None, None),
        "ɢ̆": ("uvular", "tap", None, "consonant", "voiced", None, None, None),
        "ʀ̥": ("uvular", "fricative", None, "consonant", "unvoiced", None, None, None),	
        "ʀ": ("uvular", "fricative", None, "consonant", "voiced", None, None, None),
        "ʡ": ("epi-glottal", "plosive", None, "consonant", "unvoiced", None, None, None),
        "ħ": ("epi-glottal", "fricative", None, "consonant", "unvoiced", None, None, None),
        "ʕ": ("epi-glottal", "fricative", None, "consonant", "voiced", None, None, None),
        "ʡ̆": ("epi-glottal", "tap", None, "consonant", "voiced", None, None, None),
        "ʜ": ("epi-glottal", "trill", None, "consonant", "unvoiced", None, None, None),
        "ʢ": ("epi-glottal", "trill", None, "consonant", "voiced", None, None, None),       
        "ʔ": ("glottal", "fricative", None, "consonant", "unvoiced", None, None, None),
        "h": ("glottal", "fricative", None, "consonant", "unvoiced", None, None, None),
        "ɦ": ("glottal", "fricative", None, "consonant", "voiced", None, None, None),       
        "ʔ̞": ("glottal", "approximant", None, "consonant", "voiced", None, None, None),       
        "ts": ("alveolar", "affricate", None, "consonant", "unvoiced", None, None, None),       
        "dz": ("alveolar", "affricate", None, "consonant", "voiced", None, None, None),
        "tʃ": ("post-alveolar", "affricate", None, "consonant", "unvoiced", None, None, None),       
        "dʒ": ("post-alveolar", "affricate", None, "consonant", "voiced", None, None, None),
        "ʈʂ": ("retroflex", "affricate", None, "consonant", "unvoiced", None, None, None),       
        "ɖʐ": ("retroflex", "affricate", None, "consonant", "voiced", None, None, None),
        "tɕ": ("palatal", "affricate", None, "consonant", "unvoiced", None, None, None),       
        "dʑ": ("palatal", "affricate", None, "consonant", "voiced", None, None, None),
        "i": (None, None, None, "vowel", "voiced", "high", "front", "unround"),
        "y": (None, None, None, "vowel", "voiced", "high", "front", "round"),
        "ɨ": (None, None, None, "vowel", "voiced", "high", "central", "unround"),
        "ʉ": (None, None, None, "vowel", "voiced", "high", "central", "round"),
        "ɯ": (None, None, None, "vowel", "voiced", "high", "back", "unround"),
        "u": (None, None, None, "vowel", "voiced", "high", "back", "round"),
        "ɪ": (None, None, None, "vowel", "voiced", "near-high", "front", "unround"),
        "ʏ": (None, None, None, "vowel", "voiced", "near-high", "front", "round"),
        "ʊ": (None, None, None, "vowel", "voiced", "near-high", "back", "round"),
        "e": (None, None, None, "vowel", "voiced", "mid-high", "front", "unround"),
        "ø": (None, None, None, "vowel", "voiced", "mid-high", "front", "round"),
        "ɘ": (None, None, None, "vowel", "voiced", "mid-high", "central", "unround"),
        "ɵ": (None, None, None, "vowel", "voiced", "mid-high", "central", "round"),
        "ɤ": (None, None, None, "vowel", "voiced", "mid-high", "back", "unround"),
        "o": (None, None, None, "vowel", "voiced", "mid-high", "back", "round"),
        "e̞": (None, None, None, "vowel", "voiced", "mid", "front", "unround"),
        "ø̞": (None, None, None, "vowel", "voiced", "mid", "front", "round"),
        "ə": (None, None, None, "vowel", "voiced", "mid", "central", "unround"),
        "ɤ̞": (None, None, None, "vowel", "voiced", "mid", "back", "unround"),
        "o̞": (None, None, None, "vowel", "voiced", "mid", "back", "round"),
        "ɛ": (None, None, None, "vowel", "voiced", "low-mid", "front", "unround"),
        "œ": (None, None, None, "vowel", "voiced", "low-mid", "front", "round"),
        "ɜ": (None, None, None, "vowel", "voiced", "low-mid", "central", "unround"),
        "ɞ": (None, None, None, "vowel", "voiced", "low-mid", "central", "round"),
        "ʌ": (None, None, None, "vowel", "voiced", "low-mid", "back", "unround"),
        "ɔ": (None, None, None, "vowel", "voiced", "low-mid", "back", "round"),
        "æ": (None, None, None, "vowel", "voiced", "near-low", "front", "unround"),
        "ɐ": (None, None, None, "vowel", "voiced", "near-low", "central", "unround"),
        "a": (None, None, None, "vowel", "voiced", "low", "front", "unround"),
        "ɶ": (None, None, None, "vowel", "voiced", "low", "front", "round"),
        "ä": (None, None, None, "vowel", "voiced", "low", "central", "unround"),
        "ɑ": (None, None, None, "vowel", "voiced", "low", "back", "unround"),
        "ɒ": (None, None, None, "vowel", "voiced", "low", "back", "round")
}
CHAR2VEC = {}
VEC2CHAR = {}
IDX_ORDERED = (CONSONANT_PLACES_IDX, CONSONANT_MANNERS_IDX, CONSONANT_NON_PULMONIC_IDX, CONSONANT_VOWEL_IDX, VOICING_IDX, VOWEL_HEIGHT_IDX, VOWEL_BACKNESS_IDX, VOWEL_ROUND_IDX, CONSONANT_SECOND_IDX, LENGTH_IDX)
ORDERING = ["consonant_places", "consonant_manners", "consonant_non_pulmonic", "consonant_vowel", "voicing", "vowel_height", "vowel_backness", "vowel_round", "consonant_second", "length"]

def handle_second_and_length(character: str):
    second = 1
    length = LENGTH_IDX["short"]
    if character not in CHARACTER_MAPPINGS:
        if character[-1] == 'ː':
            length = LENGTH_IDX["long"]
            character = character[:-1]
        if character not in CHARACTER_MAPPINGS:
            if character[-1] == "ˤ":
                second = CONSONANT_SECOND_IDX["pharyngealized"]
            elif character[-1] == "ʷ":
                second = CONSONANT_SECOND_IDX["labialized"]
            elif character[-1] == "ʲ":
                second = CONSONANT_SECOND_IDX["palatalized"]
            character = character[:-1]
    return second, length, character 


#NOTE: Diphthongs should be treated as seperate vowels (sketchy i know)
def ipa_to_feature_vector(character: str):
    if character not in CHAR2VEC:
        second, length, trimmed_char = handle_second_and_length(character)
        vec = list([d[x] if x is not None else 1 for d, x in zip(IDX_ORDERED[:-2], CHARACTER_MAPPINGS[trimmed_char])]) #Don't do second and length since handled seperately
        vec += [second, length]
        vec = tuple(vec)
        CHAR2VEC[character] =  vec
        VEC2CHAR[vec] = character
    return CHAR2VEC[character]

for length in ['', 'ː']:
    for second in ['', 'ˤ', 'ʷ', 'ʲ']:
        for character in CHARACTER_MAPPINGS.keys():
            if CHARACTER_MAPPINGS[character][3] == "vowel":
                ipa_to_feature_vector(character+length)
            else:
                ipa_to_feature_vector(character+second+length)
    CHAR2VEC['w'+length] = CHAR2VEC['ɰʷ'+length]
    VEC2CHAR[CHAR2VEC['w'+length]] = 'w'+length

CHAR2IDX = {k: i for i, k in enumerate(sorted(CHAR2VEC.keys()))}
IDX2CHAR = {i: k for k, i in CHAR2IDX.items()}

def feature_vector_to_ipa(vec: Tuple[str]):
    return CHAR2VEC[character]

def multilabel_ctc_log_prob(c: Dict[str, Tensor], device='cpu') -> Tensor:
    ''' Takes a dict (see PhonemeDetector) which maps phonological features to their probabilities,
    return logprob usable by CTC'''
    samples_len = c["length"].shape[0]
    batch_size = c["length"].shape[1]
    return_vector = torch.empty(samples_len, batch_size, len(IDX2CHAR)+1, device=device)
    return_vector[:, :, 0] = torch.sum(torch.stack([x[:, :, 0] for _, x in c.items()], dim=0), dim=0)
    for char_idx in range(len(IDX2CHAR)):
        vec = CHAR2VEC[IDX2CHAR[char_idx]]
        char_probs = [] 
        for i, feature in enumerate(ORDERING):
            component_prob = c[feature][:, :, vec[i]]
            char_probs.append(component_prob)
        char_probs = torch.stack(char_probs, dim=0)
        return_vector[:, :, char_idx+1] = torch.sum(char_probs, dim=0)
    return torch.nn.functional.log_softmax(return_vector, dim=-1)
