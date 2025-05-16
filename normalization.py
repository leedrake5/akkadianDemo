import re, sys, os, random, glob, typing, numpy, gc, re, requests, unicodedata,  logging, csv
import pandas as pd

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def remove_doc_refs(text: str) -> str:
    """
    1) Remove any p<digits> tokens
    2) Remove any standalone runs of 6 or more digits (e.g. '014209')
    3) Collapse leftover whitespace
    """
    # 1) strip p<digits>
    cleaned = re.sub(r'\bp\d+\b', '', text)
    # 2) strip bare doc-IDs (6+ digits)
    cleaned = re.sub(r'\b\d{6,}\b', '', cleaned)
    # 3) collapse spaces
    return re.sub(r'\s{2,}', ' ', cleaned).strip()

REMOVE_BRACKETS_TRANS = str.maketrans({
    '(': ' ',
    ')': ' ',
    '〈': ' ',
    '〉': ' ',
    '˹': ' ',
    '˺': ' ',
    '⸢': ' ',
    '⸣': ' ',
    '⌞': ' ',
    '⌟': ' ',
    '˻': ' ',
    '˼': ' ',
    '˽': ' ',
    '𐄇': ' ',
    '[': ' ',
    ']': ' ',
    '|': ' ',
    '|': ' ',
    '{': ' ',
    '}': ' ',
    '»': ' ',
    '«': ' ',
    '<': ' ',
    '>': ' ',
    '-': ' ',
    '—': ' ',
    '~': ' ',
    '+': ' ',
    '.': ' ',
    '_': ' ',
    '/': ' ',
    ',': ' ',
    ':': ' ',
    '–': ' ',
    '&': ' ',
    '!': '',
    '?': '',
    '#': '',
    '=': '',
    '%': '',
    '$': '',
    '𐄼': ''
})

def remove_brackets(s: str) -> str:
    """
    Replace various bracket-like characters (and '-') with spaces.
    """
    return s.translate(REMOVE_BRACKETS_TRANS)

def normalize_digits(s: str) -> str:
    """
    Replace any subscript or superscript digits in the input string with their 
    regular ASCII equivalents. This will convert things like "SI₂₂" or "SI²²" 
    to "SI22".
    """
    # Automatically generate mapping for subscript digits (U+2080 to U+2089).
    subscript_map = {ord(chr(0x2080 + i)): str(i) for i in range(10)}
    # Manually specify the mapping for superscript digits.
    superscript_map = {
        ord('⁰'): '0',
        ord('¹'): '1',
        ord('²'): '2',
        ord('³'): '3',
        ord('⁴'): '4',
        ord('⁵'): '5',
        ord('⁶'): '6',
        ord('⁷'): '7',
        ord('⁸'): '8',
        ord('⁹'): '9',
    }
    # Combine both mappings into one translation table.
    translation_map = {**subscript_map, **superscript_map}
    # Translate the string.
    return s.translate(translation_map)

def normalize_brackets(s: str) -> str:
    """
    Replace all types of brackets in the input string with curly brackets.
    
    The following conversions are performed:
      - Any opening bracket (e.g. '(', '[', '{', '⸢', '<', '«', '⌞') becomes '{'
      - Any closing bracket (e.g. ')', ']', '}', '⸣', '>', '»', '⌟') becomes '}'
    
    Example:
        normalize_brackets("2(u) 5(disz) i3 ak unu{ki}")
        # returns: "2{u} 5{disz} i3 ak unu{ki}"
    """
    mapping = {
        # Parentheses
        ord('('): '{', ord(')'): '}',
        # Square brackets
        ord('['): '{', ord(']'): '}',
        # Curly brackets (they might already be the desired ones, but we enforce them)
        ord('{'): '{', ord('}'): '}',
        # Alternative bracket types
        ord('⸢'): '{', ord('⸣'): '}',
        ord('<'): '{', ord('>'): '}',
        ord('«'): '{', ord('»'): '}',
        ord('⌞'): '{', ord('⌟'): '}',
    }
    return s.translate(mapping)

def gap_filler(s, source="cuneiform"):
    if source=="cuneiform":
        s = s.replace('*', ' * ')
        s = s.replace('[...]', '*')
        s = s.replace('[ ]', '*')
        s = s.replace('vac.', '*')
        s = s.replace('vac', '*')
        s = s.replace('vacat', '*')
        s = s.replace('fragmentum', '*')
        s = s.replace('fragmentum', '*')
        s = s.replace('infmut', '*')
        s = s.replace('gup', '*')
        s = s.replace('qs', '*')
        s = s.replace('vest.', '*')
        s = s.replace('vest', '*')
        s = s.replace('vestigia', '*')
        s = s.replace('...', '*')
        s = s.replace('…', '*')
        s = s.replace('. . .', '*')
        s = s.replace('xxx', '*')
        s = s.replace('x x x', '*')
        s = s.replace(' x ', ' * ')
        s = s.replace('x ', ' * ')
        s = s.replace(' x', ' * ')
        s = s.replace('($blank space$)', '*')
        s = s.replace('$blank space$', '*')
        s = s.replace('blank space', '*')
        #s = re.sub(r'x+', '<cuneiform_gap>', s)
        #s = remove_brackets(s)
        s = re.sub(r'\s+', ' ', s).strip()
    return s

def fix_cuneiform_gap(s: str) -> str:
    """
    Replaces any spaced-out version of:
      < c u n e i f o r m _ g a p >
    or
      c u n e i f o r m gap
    or
      cuneiform gap
    with <cuneiform_gap>.
    """
    pattern = re.compile(
        r'(?:<\s*c\s*u\s*n\s*e\s*i\s*f\s*o\s*r\s*m\s*_\s*g\s*a\s*p\s*>)'  # spaced-out <cuneiform_gap> with underscore
        r'|(?:c\s*u\s*n\s*e\s*i\s*f\s*o\s*r\s*m\s+gap)'                  # spaced-out c u n e i f o r m gap
        r'|(?:cuneiform\s+gap)',                                        # plain "cuneiform gap"
        flags=re.IGNORECASE
    )
    return pattern.sub("<cuneiform_gap>", s)

# Lowercase, trim, and remove non-letter characters
def normalizeString_en(s, use_prefix=False, task="Translate", target="cuneiform", type="simple", language="Akkadian", modern="English"):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?*]+", r" ", s)
    s = s.strip()
    s = remove_brackets(s)
    s = gap_filler(s)
    s = remove_doc_refs(s)
    #s = fix_cuneiform_gap(s)
    if use_prefix:
        if task=="Translate":
            if target=="cuneiform":
                return 'Translate ' + modern + ' to ' + language + ' cuneiform: ' + s
            elif target=="transliteration":
                if type == "simple":
                    return 'Translate ' + modern + ' to simple ' + language + ' transliteration: ' + s
                elif type == "group":
                    return 'Translate ' + modern + ' to grouped ' + language + ' transliteration: ' + s
                elif type == "original":
                    return 'Translate ' + modern + ' to complex ' + language + ' transliteration: ' + s
    else:
        return s


# Lowercase, trim, and remove non-letter characters
def normalizeString_cuneiform_transliterate(s, use_prefix=True, type="simple", language="Akkadian"):
    if type == "simple":
        s = unicodeToAscii(s.lower().strip())
        s = remove_brackets(s)
        s = normalize_digits(s)
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?*]+", r" ", s)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = re.sub(r'\s+', ' ', s).strip()
    elif type == "original":
        s = unicodeToAscii(s.lower().strip())
        s = remove_brackets(s)
        s = normalize_digits(s)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = re.sub(r'\s+', ' ', s).strip()
    elif type == "group":
        s = unicodeToAscii(s.lower().strip())
        s = normalize_digits(s)
        s = gap_filler(s)
        s = remove_doc_refs(s)
    normalized_string = s.strip()
    # 3) Fix spaced-out cuneiform gap tokens
    normalized_string = fix_cuneiform_gap(normalized_string)
    if use_prefix:
        if type == "simple":
            return 'Transliterate ' + language + ' cuneiform to simple Latin characters: ' + normalized_string
        elif type == "original":
            return 'Transliterate ' + language + ' cuneiform to complex Latin characters: ' + normalized_string
    else:
        return normalized_string

# Lowercase, trim, and remove non-letter characters
def normalizeString_cuneiform_rev_transliterate(s, use_prefix=True, type="simple", language="Akkadian"):
    if type == "simple":
        s = unicodeToAscii(s.lower().strip())
        s = remove_brackets(s)
        s = normalize_digits(s)
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?*]+", r" ", s)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = re.sub(r'\s+', ' ', s).strip()
    elif type == "original":
        s = unicodeToAscii(s.lower().strip())
        s = remove_brackets(s)
        s = normalize_digits(s)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = re.sub(r'\s+', ' ', s).strip()
    elif type == "group":
        s = unicodeToAscii(s.lower().strip())
        s = normalize_digits(s)
        s = gap_filler(s)
        s = remove_doc_refs(s)
    normalized_string = s.strip()
    # 3) Fix spaced-out cuneiform gap tokens
    normalized_string = fix_cuneiform_gap(normalized_string)
    if use_prefix:
        if type == "simple" :
            return 'Convert simple transliterated Latin characters to ' + language + ' cuneiform: ' + normalized_string
        elif type == "group":
            return 'Convert grouped transliterated Latin characters to ' + language + ' cuneiform: ' + normalized_string
        elif type == "original" :
            return 'Convert complex transliterated Latin characters to ' + language + ' cuneiform: ' + normalized_string
    else:
        return normalized_string

# Lowercase, trim, and remove non-letter characters
def normalizeString_cuneiform_transliterate_translate(s, use_prefix=True, task="Translate", type="simple", language="Akkadian", modern="English"):
    if type=="simple":
        s = unicodeToAscii(s.lower().strip())
        s = remove_brackets(s)
        s = normalize_digits(s)
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?*]+", r" ", s)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = re.sub(r'\s+', ' ', s).strip()
    elif type=="original":
        s = unicodeToAscii(s.lower().strip())
        s = remove_brackets(s)
        s = normalize_digits(s)
        s = gap_filler(s)
        s = remove_doc_refs(s)
        s = re.sub(r'\s+', ' ', s).strip()
    elif type == "group":
        s = unicodeToAscii(s.lower().strip())
        s = normalize_digits(s)
        s = gap_filler(s)
        s = remove_doc_refs(s)
    normalized_string = s.strip()
    # 3) Fix spaced-out cuneiform gap tokens
    normalized_string = fix_cuneiform_gap(normalized_string)
    if use_prefix:
        if task == "Translate":
            if type == "simple":
                return 'Translate simple ' + language + ' transliteration to ' + modern + ': ' + normalized_string
            elif type == "original":
                return 'Translate complex ' + language + ' transliteration to ' + modern + ': ' + normalized_string
            elif type == "group":
                return 'Translate grouped ' + language + ' transliteration to ' + modern + ': ' + normalized_string
        elif task == "Group":
            if type == "simple":
                return 'Group simple ' + language + ' transliteration into likely words: ' + normalized_string
            elif type == "original":
                return 'Group complex ' + language + ' transliteration into likely words: ' + normalized_string
    else:
        return normalized_string

# Lowercase, trim, and remove non-letter characters
def normalizeString_cuneiform_transliterate_minimal(s, use_prefix=True, language="Akkadian", modern="English"):
    s = unicodeToAscii(s.lower().strip())
    s = normalize_digits(s)
    s = gap_filler(s)
    s = remove_doc_refs(s)
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    #s = remove_brackets(s)
    normalized_string = s.strip()
    # 3) Fix spaced-out cuneiform gap tokens
    #normalized_string = fix_cuneiform_gap(normalized_string)
    if use_prefix:
        return 'Translate ' + language + ' grouped transliteration to ' + modern + ': ' + normalized_string
    else:
        return normalized_string

def fix_suprasigillum(text):
    # This pattern means:
    #  - 's' followed by 1+ non-word characters (spaces, etc.)
    #  - 'u' followed by 1+ non-word chars
    #  - 'p' ...
    #  - and so on until 'm'
    pattern = r"s\W+u\W+p\W+r\W+a\W+s\W+i\W+g\W+i\W+l\W+l\W+u\W+m"
    return re.sub(pattern, "suprasigillum", text)


def normalizeString_cuneiform(s, use_prefix=True, task="Translate", type="simple", language="Akkadian", modern="English"):
    # Optional: Remove unwanted modern characters, if any (adjust regex as needed)
    # s = re.sub(r'[^\u12000-\u123FF\u12400-\u1247F]+', '', s)  # Adjust Unicode ranges to cuneiform and related characters
    # Split each character/sign into separate entries
    # This assumes each character in the string is a distinct sign, no need to join with spaces if already separated
    s = remove_brackets(s)
    s = gap_filler(s)
    s = remove_doc_refs(s)
    normalized_string = ' '.join(s)  # This joins every character with a space, treating each as a separate token
    normalized_string = fix_suprasigillum(normalized_string)
    # Add the prefix if use_prefix is True
    if use_prefix:
        if task == "Translate":
            return 'Translate ' + language + ' cuneiform to '  + modern + ': ' + normalized_string
        elif task == "Transliterate":
            if type == "simple":
                return 'Transliterate ' + language + ' cuneiform to simple Latin characters: ' + normalized_string
            elif type == "group":
                return 'Transliterate ' + language + ' cuneiform to grouped Latin characters: ' + normalized_string
            elif type == "original":
                return 'Transliterate ' + language + ' cuneiform to complex Latin characters: ' + normalized_string
    else:
        return normalized_string

def read_and_process_file(file_path):
    # Check if the file_path is a URL
    if file_path.startswith('http://') or file_path.startswith('https://'):
        # Fetch the content from the URL
        response = requests.get(file_path)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        lines = response.text.strip().split('\n')
    else:
        # Open the local file and read the lines
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().strip().split('\n')
    # Replace ". . . " with "*" in each line
    processed_lines = [re.sub(r'\s*\.\s*\.\s*\.\s*', '*', line) for line in lines]
    return processed_lines

def convert(lst):
   res_dict = {}
   for i in range(0, len(lst), 2):
       res_dict[lst[i]] = lst[i + 1]
   return res_dict

def collapse_spaces(obj):
    # This function now does no printing.
    # Just processes the string(s) and returns them silently.
    def _collapse_spaces_in_string(s):
        return re.sub(r'\s+', ' ', s).strip()
    if isinstance(obj, str):
        return _collapse_spaces_in_string(obj)
    elif isinstance(obj, (tuple, list)) and len(obj) == 2 and all(isinstance(x, str) for x in obj):
        return (
            _collapse_spaces_in_string(obj[0]),
            _collapse_spaces_in_string(obj[1])
        )
    else:
        raise ValueError(f"Expected a single string or a 2-string pair, got: {obj}")

import re
import unicodedata

def remove_control_characters(s):
    """
    Remove all Cc, Cf, Cs, Co, Cn categories — i.e. non-printable/control chars.
    """
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

def normalize(text):
    # 1. Remove control characters (not just ASCII)
    text = remove_control_characters(text)
    # 2. Replace consecutive whitespace with a single space
    text = re.sub(r"\s+", " ", text)
    # 3. Trim
    text = text.strip()
    return text

def trim_singles(pairs, max_length1, max_length2, max_length_threshold, min_length_threshold):
    # 1. Filter out any pair where pair is None or pair[0] is None
    valid_pairs = []
    for pair in pairs:
        # Make sure the pair itself is not None AND has at least one element
        if pair and pair[0]:
            valid_pairs.append(pair)
    # 2. Filter out pairs by word count threshold
    max_filtered_pairs = [
        p for p in valid_pairs
        if len(p[0].split()) <= max_length_threshold
    ]
    min_filtered_pairs = [
        p for p in max_filtered_pairs
        if len(p[0].split()) >= min_length_threshold
    ]
    # 3. Normalize and trim
    trimmed_pairs = []
    for p in min_filtered_pairs:
        s1 = p[0]  # p is presumably a 1-element tuple or a 1-element list
        # Normalize
        s1_norm = normalize(s1)
        # Truncate
        s1_trunc = s1_norm[:max_length1]
        # Normalize again just to be safe
        s1_final = normalize(s1_trunc)
        trimmed_pairs.append(s1_final)
    return trimmed_pairs

def trim_pairs(pairs, max_length1, max_length2, max_length_threshold, min_length_threshold):
    valid_pairs = []
    for pair in pairs:
        # Ensure the pair has 2 elements and neither is None
        if pair and len(pair) == 2 and pair[0] and pair[1]:
            valid_pairs.append(pair)
    # Filter out pairs by word count threshold
    max_filtered_pairs = [
        (s1, s2) for s1, s2 in valid_pairs
        if len(s1.split()) <= max_length_threshold and len(s2.split()) <= (max_length_threshold - 5)
    ]
    min_filtered_pairs = [
        (s1, s2) for s1, s2 in max_filtered_pairs
        if len(s1.split()) >= min_length_threshold and len(s2.split()) >= min_length_threshold
    ]
    trimmed_pairs = []
    for s1, s2 in min_filtered_pairs:
        # Normalize
        s1_norm = normalize(s1)
        s2_norm = normalize(s2)
        # Truncate
        s1_trunc = s1_norm[:max_length1]
        s2_trunc = s2_norm[:max_length2]
        # Normalize again
        s1_final = normalize(s1_trunc)
        s2_final = normalize(s2_trunc)
        trimmed_pairs.append((s1_final, s2_final))
    return trimmed_pairs
