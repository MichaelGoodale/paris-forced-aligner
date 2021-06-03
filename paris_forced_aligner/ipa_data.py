from typing import Tuple, Dict

import torch 
import numpy as np
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
CONSONANT_SECOND = ["oral", "labialized", "pharyngealized", "palatalized"]
VOICING = ["voiced", "unvoiced"]
VOWEL_HEIGHT = ["high", "near-high", "mid-high", "mid", "low-mid", "near-low", "low"]
VOWEL_BACKNESS = ["front", "central", "back"]
VOWEL_ROUND = ["round", "unround"]
LENGTH = ["short", "long"]

#idx=0 for blank, idx=1 for when the feature is not relevant
CONSONANT_PLACES_IDX = {x: i for i, x in enumerate(CONSONANT_PLACES)}
CONSONANT_MANNERS_IDX = {x: i for i, x in enumerate(CONSONANT_MANNERS)}
CONSONANT_SECOND_IDX = {x: i for i, x in enumerate(CONSONANT_SECOND)}
VOICING_IDX = {x: i for i, x in enumerate(VOICING)}
VOWEL_HEIGHT_IDX = {x: i for i, x in enumerate(VOWEL_HEIGHT)}
VOWEL_BACKNESS_IDX = {x: i for i, x in enumerate(VOWEL_BACKNESS)}
VOWEL_ROUND_IDX = {x: i for i, x in enumerate(VOWEL_ROUND)}
LENGTH_IDX = {x: i for i, x in enumerate(LENGTH)}

VOCAB_SIZES = {"consonant_places": len(CONSONANT_PLACES),
        "consonant_manners": len(CONSONANT_MANNERS),
        "consonant_second": len(CONSONANT_SECOND),
        "voicing": len(VOICING),
        "vowel_height": len(VOWEL_HEIGHT),
        "vowel_backness": len(VOWEL_BACKNESS),
        "vowel_round": len(VOWEL_ROUND),
        "length": len(LENGTH),
        "consonant_vowel": 2,
        "blank": 2}

CHARACTER_MAPPINGS = {
        "m": ("bilabial", "nasal", "voiced", None, None, None),
        "p": ("bilabial", "plosive", "unvoiced", None, None, None),
        "b": ("bilabial", "plosive", "voiced", None, None, None),
        "ɸ": ("bilabial", "fricative", "unvoiced", None, None, None),
        "β": ("bilabial", "fricative", "voiced", None, None, None),
        "ⱱ̟": ("bilabial", "tap", "voiced", None, None, None),
        "ʙ̥": ("bilabial", "trill", "unvoiced", None, None, None),
        "ʙ": ("bilabial", "trill", "voiced", None, None, None),
        "ɱ": ("labio-dental", "nasal", "voiced", None, None, None),
        "p̪": ("labio-dental", "plosive", "unvoiced", None, None, None),
        "b̪": ("labio-dental", "plosive", "voiced", None, None, None),
        "f": ("labio-dental", "fricative", "unvoiced", None, None, None),
        "v": ("labio-dental", "fricative", "voiced", None, None, None),
        "ʋ": ("labio-dental", "approximant", "voiced", None, None, None),
        "ⱱ": ("labio-dental", "tap", "voiced", None, None, None),
        "n̥": ("alveolar", "nasal", "unvoiced", None, None, None),
        "n": ("alveolar", "nasal", "voiced", None, None, None),
        "t": ("alveolar", "plosive", "unvoiced", None, None, None),
        "d": ("alveolar", "plosive", "voiced", None, None, None),
        "s": ("alveolar", "sibilant", "unvoiced", None, None, None),
        "z": ("alveolar", "sibilant", "voiced", None, None, None),
        "ʃ": ("post-alveolar", "sibilant", "unvoiced", None, None, None),
        "ʒ": ("post-alveolar", "sibilant", "voiced", None, None, None),
        "t̪": ("dental", "plosive", "unvoiced", None, None, None),
        "d̪": ("dental", "plosive", "voiced", None, None, None),
        "θ": ("dental", "fricative", "unvoiced", None, None, None),
        "ð": ("dental", "fricative", "voiced", None, None, None),
        "ɹ": ("alveolar", "approximant", "voiced", None, None, None),
        "ɾ̥": ("alveolar", "tap", "unvoiced", None, None, None),
        "ɾ": ("alveolar", "tap", "voiced", None, None, None),
        "r̥": ("alveolar", "trill", "unvoiced", None, None, None),
        "r": ("alveolar", "trill", "voiced", None, None, None),
        "ɬ": ("alveolar", "lateral fricative", "unvoiced", None, None, None),
        "ɮ": ("alveolar", "lateral fricative", "voiced", None, None, None),
        "l": ("alveolar", "lateral approximant", "unvoiced", None, None, None),
        "ɺ̥": ("alveolar", "lateral tap", "unvoiced", None, None, None),
        "ɺ": ("alveolar", "lateral tap", "voiced", None, None, None),
        "ɳ̊": ("retroflex", "nasal", "unvoiced", None, None, None),
        "ɳ": ("retroflex", "nasal", "voiced", None, None, None),
        "ʈ": ("retroflex", "plosive", "unvoiced", None, None, None),
        "ɖ": ("retroflex", "plosive", "voiced", None, None, None),
        "ʂ": ("retroflex", "sibilant", "unvoiced", None, None, None),
        "ʐ": ("retroflex", "sibilant", "voiced", None, None, None),
        "ɻ": ("retroflex", "approximant", "voiced", None, None, None),
        "ɭ": ("retroflex", "lateral approximant", "voiced", None, None, None),
        "ɭ̥̆": ("retroflex", "lateral tap", "unvoiced", None, None, None),
        "ɭ̆": ("retroflex", "lateral tap", "voiced", None, None, None),
        "ɲ̊": ("palatal", "nasal", "unvoiced", None, None, None),
        "ɲ": ("palatal", "nasal", "voiced", None, None, None),
        "c": ("palatal", "plosive", "unvoiced", None, None, None),
        "ɟ": ("palatal", "plosive", "voiced", None, None, None),
        "ɕ": ("palatal", "sibilant", "unvoiced", None, None, None),
        "ʑ": ("palatal", "sibilant", "voiced", None, None, None),
        "ç": ("palatal", "fricative", "unvoiced", None, None, None),
        "ʝ": ("palatal", "fricative", "voiced", None, None, None),
        "j": ("palatal", "approximant", "voiced", None, None, None),
        "ʎ": ("palatal", "lateral approximant", "voiced", None, None, None),
        "ŋ̊": ("velar", "nasal", "unvoiced", None, None, None),
        "ŋ": ("velar", "nasal", "voiced", None, None, None),
        "k": ("velar", "plosive", "unvoiced", None, None, None),
        "g": ("velar", "plosive", "voiced", None, None, None),
        "x": ("velar", "fricative", "unvoiced", None, None, None),
        "ɣ": ("velar", "fricative", "voiced", None, None, None),
        "ɰ": ("velar", "approximant", "voiced", None, None, None),
        "ɴ": ("uvular", "nasal", "voiced", None, None, None),
        "q": ("uvular", "plosive", "unvoiced", None, None, None),
        "ɢ": ("uvular", "plosive", "voiced", None, None, None),
        "χ": ("uvular", "fricative", "unvoiced", None, None, None),
        "ʁ": ("uvular", "fricative", "voiced", None, None, None),
        "ɢ̆": ("uvular", "tap", "voiced", None, None, None),
        "ʀ̥": ("uvular", "trill", "unvoiced", None, None, None),
        "ʀ": ("uvular", "trill", "voiced", None, None, None),
        "ʡ": ("epi-glottal", "plosive", "unvoiced", None, None, None),
        "ħ": ("epi-glottal", "fricative", "unvoiced", None, None, None),
        "ʕ": ("epi-glottal", "fricative", "voiced", None, None, None),
        "ʡ̆": ("epi-glottal", "tap", "voiced", None, None, None),
        "ʜ": ("epi-glottal", "trill", "unvoiced", None, None, None),
        "ʢ": ("epi-glottal", "trill", "voiced", None, None, None),
        "ʔ": ("glottal", "fricative", "unvoiced", None, None, None),
        "h": ("glottal", "fricative", "unvoiced", None, None, None),
        "ɦ": ("glottal", "fricative", "voiced", None, None, None),
        "ʔ̞": ("glottal", "approximant", "voiced", None, None, None),
        "ts": ("alveolar", "affricate", "unvoiced", None, None, None),
        "dz": ("alveolar", "affricate", "voiced", None, None, None),
        "tʃ": ("post-alveolar", "affricate", "unvoiced", None, None, None),
        "dʒ": ("post-alveolar", "affricate", "voiced", None, None, None),
        "ʈʂ": ("retroflex", "affricate", "unvoiced", None, None, None),
        "ɖʐ": ("retroflex", "affricate", "voiced", None, None, None),
        "tɕ": ("palatal", "affricate", "unvoiced", None, None, None),
        "dʑ": ("palatal", "affricate", "voiced", None, None, None),
        "i": (None, None, "voiced", "high", "front", "unround"),
        "y": (None, None, "voiced", "high", "front", "round"),
        "ɨ": (None, None, "voiced", "high", "central", "unround"),
        "ʉ": (None, None, "voiced", "high", "central", "round"),
        "ɯ": (None, None, "voiced", "high", "back", "unround"),
        "u": (None, None, "voiced", "high", "back", "round"),
        "ɪ": (None, None, "voiced", "near-high", "front", "unround"),
        "ʏ": (None, None, "voiced", "near-high", "front", "round"),
        "ʊ": (None, None, "voiced", "near-high", "back", "round"),
        "e": (None, None, "voiced", "mid-high", "front", "unround"),
        "ø": (None, None, "voiced", "mid-high", "front", "round"),
        "ɘ": (None, None, "voiced", "mid-high", "central", "unround"),
        "ɵ": (None, None, "voiced", "mid-high", "central", "round"),
        "ɤ": (None, None, "voiced", "mid-high", "back", "unround"),
        "o": (None, None, "voiced", "mid-high", "back", "round"),
        "e̞": (None, None, "voiced", "mid", "front", "unround"),
        "ø̞": (None, None, "voiced", "mid", "front", "round"),
        "ə": (None, None, "voiced", "mid", "central", "unround"),
        "ɤ̞": (None, None, "voiced", "mid", "back", "unround"),
        "o̞": (None, None, "voiced", "mid", "back", "round"),
        "ɛ": (None, None, "voiced", "low-mid", "front", "unround"),
        "œ": (None, None, "voiced", "low-mid", "front", "round"),
        "ɜ": (None, None, "voiced", "low-mid", "central", "unround"),
        "ɞ": (None, None, "voiced", "low-mid", "central", "round"),
        "ʌ": (None, None, "voiced", "low-mid", "back", "unround"),
        "ɔ": (None, None, "voiced", "low-mid", "back", "round"),
        "æ": (None, None, "voiced", "near-low", "front", "unround"),
        "ɐ": (None, None, "voiced", "near-low", "central", "unround"),
        "a": (None, None, "voiced", "low", "front", "unround"),
        "ɶ": (None, None, "voiced", "low", "front", "round"),
        "ä": (None, None, "voiced", "low", "central", "unround"),
        "ɑ": (None, None, "voiced", "low", "back", "unround"),
        "ɒ": (None, None, "voiced", "low", "back", "round")
}
VECTOR_MAPPINGS = {v: k for k, v in CHARACTER_MAPPINGS.items()}
IDX_ORDERED = (CONSONANT_PLACES_IDX, CONSONANT_MANNERS_IDX, VOICING_IDX, VOWEL_HEIGHT_IDX, VOWEL_BACKNESS_IDX, VOWEL_ROUND_IDX, CONSONANT_SECOND_IDX, LENGTH_IDX)
ORDERING = ["consonant_vowel", "consonant_places", "consonant_manners", "voicing", "vowel_height", "vowel_backness", "vowel_round", "consonant_second", "length"]
SECOND_MAPPER = {"oral": '', "labialized":'ʷ', "pharyngealized": 'ˤ', "palatalized": 'ʲ'}
LENGTH_MAPPER = {"long": 'ː', "short": ''}
IDX2CHAR = {}
CHAR2IDX = {}

def vector_generator():
    vector_idx = 1
    for place, place_idx in CONSONANT_PLACES_IDX.items():
        for manner, manner_idx in CONSONANT_MANNERS_IDX.items():
            for voicing, voicing_idx in VOICING_IDX.items():
                consonant_vector = (place, manner, voicing, None, None, None)
                if consonant_vector in VECTOR_MAPPINGS:
                    base_char = VECTOR_MAPPINGS[consonant_vector]
                else:
                    base_char = None #Nonarticutable / not written
                for second, second_idx in CONSONANT_SECOND_IDX.items():
                    for length, length_idx in LENGTH_IDX.items():
                        if base_char is not None:
                            IDX2CHAR[vector_idx] = base_char + SECOND_MAPPER[second] + LENGTH_MAPPER[length]
                            CHAR2IDX[IDX2CHAR[vector_idx]] = vector_idx
                        yield vector_idx, (1, place_idx, manner_idx, voicing_idx, None, None, None, second_idx, length_idx)
                        vector_idx += 1

    for height, height_idx in VOWEL_HEIGHT_IDX.items():
        for back, back_idx in VOWEL_BACKNESS_IDX.items():
            for vowel_round, vowel_round_idx in VOWEL_ROUND_IDX.items():
                for voicing, voicing_idx in VOICING_IDX.items():
                    consonant_vector = (None, None, voicing, height, back, vowel_round)
                    if consonant_vector in VECTOR_MAPPINGS:
                        base_char = VECTOR_MAPPINGS[consonant_vector]
                    else:
                        base_char = None #Nonarticutable / not written
                    for length, length_idx in LENGTH_IDX.items():
                        if base_char is not None:
                            IDX2CHAR[vector_idx] = base_char + LENGTH_MAPPER[length]
                            CHAR2IDX[IDX2CHAR[vector_idx]] = vector_idx
                        yield vector_idx, (0, None, None, voicing_idx, height_idx, back_idx, vowel_round_idx, None, length_idx)
                        vector_idx += 1

VECTORS = [(idx, vec) for idx, vec in vector_generator()]
COLUMNS = {k: [-1] for k in ORDERING}

for i, vec in VECTORS:
    for v, feature in enumerate(ORDERING):
        COLUMNS[feature].append(vec[v] if vec[v] is not None else -1)

for k in COLUMNS:
    COLUMNS[k] = np.array(COLUMNS[k])

CHAR2IDX['w'] = CHAR2IDX["ɰʷ"] 
CHAR2IDX['wː'] = CHAR2IDX["ɰʷː"] 
VOCAB_SIZE = len(VECTORS) + 1

def old_multilabel_ctc_log_prob(c, device='cpu'):
    ''' Takes a dict (see PhonemeDetector) which maps phonological features to their probabilities,
    return logprob usable by CTC'''
    samples_len = c["length"].shape[0]
    batch_size = c["length"].shape[1]
    return_vector = torch.empty(samples_len, batch_size, VOCAB_SIZE, device=device)
    return_vector[:, :, 0] = c['blank'][:, :, 0]
    for char_idx, vec in VECTORS:
        char_probs = [c['blank'][:, :, 1]]
        for i, feature in enumerate(ORDERING):
            if vec[i] is not None:
                component_prob = c[feature][:, :, vec[i]]
                char_probs.append(component_prob)
        char_probs = torch.stack(char_probs, dim=0)
        return_vector[:, :, char_idx] = torch.sum(char_probs, dim=0)
    return return_vector

def multilabel_ctc_log_prob(c: Dict[str, Tensor], device='cpu') -> Tensor:
    ''' Takes a dict (see PhonemeDetector) which maps phonological features to their probabilities,
    return logprob usable by CTC'''
    samples_len = c["length"].shape[0]
    batch_size = c["length"].shape[1]
    return_vector = torch.zeros(samples_len, batch_size, VOCAB_SIZE, device=device)
    return_vector[:, :, 0] = c['blank'][:, :, 0]
    return_vector[:, :, 1:] = c['blank'][:, :, 1].unsqueeze(-1).expand(samples_len, batch_size, VOCAB_SIZE - 1)
    for feature, column in COLUMNS.items():
        for i in range(VOCAB_SIZES[feature]):
            return_vector[:, :, column == i] += c[feature][:, :, i].unsqueeze(-1)
    return return_vector
