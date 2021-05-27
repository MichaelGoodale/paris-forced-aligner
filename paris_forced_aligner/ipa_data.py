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
CONSONANT_PLACES_IDX = {x:i+2 for i, x in enumerate(CONSONANT_PLACES)}
CONSONANT_MANNERS_IDX = {x:i+2 for i, x in enumerate(CONSONANT_MANNERS)}
CONSONANT_SECOND_IDX = {x:i+2 for i, x in enumerate(CONSONANT_SECOND)}
CONSONANT_NON_PULMONIC_IDX = {x:i+2 for i, x in enumerate(CONSONANT_PLACES)}
CONSONANT_VOWEL_IDX = {x:i+1 for i, x in enumerate(CONSONANT_VOWEL)}
VOICING_IDX = {x:i+1 for i, x in enumerate(VOICING)}

VOWEL_HEIGHT_IDX = {x:i+2 for i, x in enumerate(VOWEL_HEIGHT)}
VOWEL_BACKNESS_IDX = {x:i+2 for i, x in enumerate(VOWEL_BACKNESS)}
VOWEL_ROUND_IDX = {x:i+2 for i, x in enumerate(VOWEL_ROUND)}

LENGTH_IDX = {x:i+1 for i, x in enumerate(LENGTH)}

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

CHARACTER_MAPPINGS = {
        "m": ("bilabial", "nasal", 0, "consonant", "voiced", 0, 0, 0),
        "p": ("bilabial", "plosive", 0, "consonant", "unvoiced", 0, 0, 0),
        "b": ("bilabial", "plosive", 0, "consonant", "voiced", 0, 0, 0),
        "ɸ": ("bilabial", "fricative", 0, "consonant", "unvoiced", 0, 0, 0),
        "β": ("bilabial", "fricative", 0, "consonant", "voiced", 0, 0, 0),
        "ⱱ̟": ("bilabial", "tap", 0, "consonant", "voiced", 0, 0, 0),
        "ʙ̥": ("bilabial", "trill", 0, "consonant", "unvoiced", 0, 0, 0),
        "ʙ": ("bilabial", "trill", 0, "consonant", "voiced", 0, 0, 0),
        "ɱ": ("labio-dental", "nasal", 0, "consonant", "voiced", 0, 0, 0),
        "p̪": ("labio-dental", "plosive", 0, "consonant", "unvoiced", 0, 0, 0),
        "b̪": ("labio-dental", "plosive", 0, "consonant", "voiced", 0, 0, 0),
        "f": ("labio-dental", "fricative", 0, "consonant", "unvoiced", 0, 0, 0),
        "v": ("labio-dental", "fricative", 0, "consonant", "voiced", 0, 0, 0),
        "ʋ": ("labio-dental", "approximant", 0, "consonant", "voiced", 0, 0, 0),
        "ⱱ": ("labio-dental", "tap", 0, "consonant", "voiced", 0, 0, 0),
        "n̥": ("alveolar", "nasal", 0, "consonant", "unvoiced", 0, 0, 0),
        "n": ("alveolar", "nasal", 0, "consonant", "voiced", 0, 0, 0),
        "t": ("alveolar", "plosive", 0, "consonant", "unvoiced", 0, 0, 0),
        "d": ("alveolar", "plosive", 0, "consonant", "voiced", 0, 0, 0),
        "s": ("alveolar", "sibilant", 0, "consonant", "unvoiced", 0, 0, 0),
        "z": ("alveolar", "sibilant", 0, "consonant", "voiced", 0, 0, 0),
        "ʃ": ("post-alveolar", "sibilant", 0, "consonant", "unvoiced", 0, 0, 0),
        "ʒ": ("post-alveolar", "sibilant", 0, "consonant", "voiced", 0, 0, 0),
        "t̪": ("dental", "plosive", 0, "consonant", "unvoiced", 0, 0, 0),
        "d̪": ("dental", "plosive", 0, "consonant", "voiced", 0, 0, 0),
        "θ": ("dental", "fricative", 0, "consonant", "unvoiced", 0, 0, 0),
        "ð": ("dental", "fricative", 0, "consonant", "voiced", 0, 0, 0),
        "ɹ": ("alveolar", "approximant", 0, "consonant", "voiced", 0, 0, 0),
        "ɾ̥": ("alveolar", "tap", 0, "consonant", "unvoiced", 0, 0, 0),
        "ɾ": ("alveolar", "tap", 0, "consonant", "voiced", 0, 0, 0),
        "r̥": ("alveolar", "trill", 0, "consonant", "unvoiced", 0, 0, 0),
        "r": ("alveolar", "trill", 0, "consonant", "voiced", 0, 0, 0),
        "ɬ": ("alveolar", "lateral fricative", 0, "consonant", "unvoiced", 0, 0, 0),
        "ɮ": ("alveolar", "lateral fricative", 0, "consonant", "voiced", 0, 0, 0),   	
        "l": ("alveolar", "lateral approximant", 0, "consonant", "unvoiced", 0, 0, 0),
        "ɺ̥": ("alveolar", "lateral tap", 0, "consonant", "unvoiced", 0, 0, 0),
        "ɺ": ("alveolar", "lateral tap", 0, "consonant", "voiced", 0, 0, 0),
        "ɳ̊": ("retroflex", "nasal", 0, "consonant", "unvoiced", 0, 0, 0),
        "ɳ": ("retroflex", "nasal", 0, "consonant", "voiced", 0, 0, 0),
        "ʈ": ("retroflex", "plosive", 0, "consonant", "unvoiced", 0, 0, 0),
        "ɖ": ("retroflex", "plosive", 0, "consonant", "voiced", 0, 0, 0),
        "ʂ": ("retroflex", "sibilant", 0, "consonant", "unvoiced", 0, 0, 0),
        "ʐ": ("retroflex", "sibilant", 0, "consonant", "voiced", 0, 0, 0),
        "ɻ": ("retroflex", "approximant", 0, "consonant", "voiced", 0, 0, 0),
        "ɭ": ("retroflex", "lateral approximant", 0, "consonant", "voiced", 0, 0, 0),
        "ɭ̥̆": ("retroflex", "lateral tap", 0, "consonant", "unvoiced", 0, 0, 0),
        "ɭ̆": ("retroflex", "lateral tap", 0, "consonant", "voiced", 0, 0, 0),
        "ɲ̊": ("palatal", "nasal", 0, "consonant", "unvoiced", 0, 0, 0),
        "ɲ": ("palatal", "nasal", 0, "consonant", "voiced", 0, 0, 0),
        "c": ("palatal", "plosive", 0, "consonant", "unvoiced", 0, 0, 0),
        "ɟ": ("palatal", "plosive", 0, "consonant", "voiced", 0, 0, 0),
        "ɕ": ("palatal", "sibilant", 0, "consonant", "unvoiced", 0, 0, 0),
        "ʑ": ("palatal", "sibilant", 0, "consonant", "voiced", 0, 0, 0),
        "ç": ("palatal", "fricative", 0, "consonant", "unvoiced", 0, 0, 0),
        "ʝ": ("palatal", "fricative", 0, "consonant", "voiced", 0, 0, 0),
        "j": ("palatal", "approximant", 0, "consonant", "voiced", 0, 0, 0),
        "ʎ": ("palatal", "lateral approximant", 0, "consonant", "voiced", 0, 0, 0),
        "ŋ̊": ("velar", "nasal", 0, "consonant", "unvoiced", 0, 0, 0),
        "ŋ": ("velar", "nasal", 0, "consonant", "voiced", 0, 0, 0), 	
        "k": ("velar", "plosive", 0, "consonant", "unvoiced", 0, 0, 0),
        "g": ("velar", "plosive", 0, "consonant", "voiced", 0, 0, 0),
        "x": ("velar", "fricative", 0, "consonant", "unvoiced", 0, 0, 0),
        "ɣ": ("velar", "fricative", 0, "consonant", "voiced", 0, 0, 0),
        "ɰ": ("velar", "approximant", 0, "consonant", "voiced", 0, 0, 0),
        "ɴ": ("uvular", "nasal", 0, "consonant", "voiced", 0, 0, 0), 	
        "q": ("uvular", "plosive", 0, "consonant", "unvoiced", 0, 0, 0),
        "ɢ": ("uvular", "plosive", 0, "consonant", "voiced", 0, 0, 0),
        "χ": ("uvular", "fricative", 0, "consonant", "unvoiced", 0, 0, 0),
        "ʁ": ("uvular", "fricative", 0, "consonant", "voiced", 0, 0, 0),
        "ɢ̆": ("uvular", "tap", 0, "consonant", "voiced", 0, 0, 0),
        "ʀ̥": ("uvular", "fricative", 0, "consonant", "unvoiced", 0, 0, 0),	
        "ʀ": ("uvular", "fricative", 0, "consonant", "voiced", 0, 0, 0),
        "ʡ": ("epi-glottal", "plosive", 0, "consonant", "unvoiced", 0, 0, 0),
        "ħ": ("epi-glottal", "fricative", 0, "consonant", "unvoiced", 0, 0, 0),
        "ʕ": ("epi-glottal", "fricative", 0, "consonant", "voiced", 0, 0, 0),
        "ʡ̆": ("epi-glottal", "tap", 0, "consonant", "voiced", 0, 0, 0),
        "ʜ": ("epi-glottal", "trill", 0, "consonant", "unvoiced", 0, 0, 0),
        "ʢ": ("epi-glottal", "trill", 0, "consonant", "voiced", 0, 0, 0),       
        "ʔ": ("glottal", "fricative", 0, "consonant", "unvoiced", 0, 0, 0),
        "h": ("glottal", "fricative", 0, "consonant", "unvoiced", 0, 0, 0),
        "ɦ": ("glottal", "fricative", 0, "consonant", "voiced", 0, 0, 0),       
        "ʔ̞": ("glottal", "approximant", 0, "consonant", "voiced", 0, 0, 0),       
        "ts": ("alveolar", "affricate", 0, "consonant", "unvoiced", 0, 0, 0),       
        "dz": ("alveolar", "affricate", 0, "consonant", "voiced", 0, 0, 0),
        "tʃ": ("post-alveolar", "affricate", 0, "consonant", "unvoiced", 0, 0, 0),       
        "dʒ": ("post-alveolar", "affricate", 0, "consonant", "voiced", 0, 0, 0),
        "ʈʂ": ("retroflex", "affricate", 0, "consonant", "unvoiced", 0, 0, 0),       
        "ɖʐ": ("retroflex", "affricate", 0, "consonant", "voiced", 0, 0, 0),
        "tɕ": ("palatal", "affricate", 0, "consonant", "unvoiced", 0, 0, 0),       
        "dʑ": ("palatal", "affricate", 0, "consonant", "voiced", 0, 0, 0),
        "i": (0, 0, 0, "vowel", "voiced", "high", "front", "unround"),
        "y": (0, 0, 0, "vowel", "voiced", "high", "front", "round"),
        "ɨ": (0, 0, 0, "vowel", "voiced", "high", "central", "unround"),
        "ʉ": (0, 0, 0, "vowel", "voiced", "high", "central", "round"),
        "ɯ": (0, 0, 0, "vowel", "voiced", "high", "back", "unround"),
        "u": (0, 0, 0, "vowel", "voiced", "high", "back", "round"),
        "ɪ": (0, 0, 0, "vowel", "voiced", "near-high", "front", "unround"),
        "ʏ": (0, 0, 0, "vowel", "voiced", "near-high", "front", "round"),
        "ʊ": (0, 0, 0, "vowel", "voiced", "near-high", "back", "round"),
        "e": (0, 0, 0, "vowel", "voiced", "mid-high", "front", "unround"),
        "ø": (0, 0, 0, "vowel", "voiced", "mid-high", "front", "round"),
        "ɘ": (0, 0, 0, "vowel", "voiced", "mid-high", "central", "unround"),
        "ɵ": (0, 0, 0, "vowel", "voiced", "mid-high", "central", "round"),
        "ɤ": (0, 0, 0, "vowel", "voiced", "mid-high", "back", "unround"),
        "o": (0, 0, 0, "vowel", "voiced", "mid-high", "back", "round"),
        "e̞": (0, 0, 0, "vowel", "voiced", "mid", "front", "unround"),
        "ø̞": (0, 0, 0, "vowel", "voiced", "mid", "front", "round"),
        "ə": (0, 0, 0, "vowel", "voiced", "mid", "central", "unround"),
        "ɤ̞": (0, 0, 0, "vowel", "voiced", "mid", "back", "unround"),
        "o̞": (0, 0, 0, "vowel", "voiced", "mid", "back", "round"),
        "ɛ": (0, 0, 0, "vowel", "voiced", "low-mid", "front", "unround"),
        "œ": (0, 0, 0, "vowel", "voiced", "low-mid", "front", "round"),
        "ɜ": (0, 0, 0, "vowel", "voiced", "low-mid", "central", "unround"),
        "ɞ": (0, 0, 0, "vowel", "voiced", "low-mid", "central", "round"),
        "ʌ": (0, 0, 0, "vowel", "voiced", "low-mid", "back", "unround"),
        "ɔ": (0, 0, 0, "vowel", "voiced", "low-mid", "back", "round"),
        "æ": (0, 0, 0, "vowel", "voiced", "near-low", "front", "unround"),
        "ɐ": (0, 0, 0, "vowel", "voiced", "near-low", "central", "unround"),
        "a": (0, 0, 0, "vowel", "voiced", "low", "front", "unround"),
        "ɶ": (0, 0, 0, "vowel", "voiced", "low", "front", "round"),
        "ä": (0, 0, 0, "vowel", "voiced", "low", "central", "unround"),
        "ɑ": (0, 0, 0, "vowel", "voiced", "low", "back", "unround"),
        "ɒ": (0, 0, 0, "vowel", "voiced", "low", "back", "round")
}
CHAR2VEC = {}
VEC2CHAR = {}
IDX_ORDERED = (CONSONANT_PLACES_IDX, CONSONANT_MANNERS_IDX, CONSONANT_NON_PULMONIC_IDX, CONSONANT_VOWEL_IDX, VOICING_IDX, VOWEL_HEIGHT_IDX, VOWEL_BACKNESS_IDX, VOWEL_ROUND_IDX, CONSONANT_SECOND_IDX, LENGTH_IDX)
ORDERING = ["consonant_places", "consonant_manners", "consonant_non_pulmonic", "consonant_vowel", "voicing", "vowel_height", "vowel_backness", "vowel_round", "consonant_second", "length"]

def handle_second_and_length(character: str):
    second = 0
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
        vec = list([d[x] if x != 0 else 0 for d, x in zip(IDX_ORDERED[:-2], CHARACTER_MAPPINGS[trimmed_char])]) #Don't do second and length since handled seperately
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
            if vec[i] == 1: #(component blank):
                component_prob = 0*component_prob
            char_probs.append(component_prob)
        char_probs = torch.stack(char_probs, dim=0)
        return_vector[:, :, char_idx+1] = torch.sum(char_probs, dim=0)
    return return_vector
