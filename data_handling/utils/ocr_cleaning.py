"""
'ocr_cleaning' module for ocr transcript cleaning
"""
import re

class OCR_Clean:
    """
    OCR_Clean class contains functions to clean OCR scanned strings, primarly ultizing and abstracting 
    regex functionality, creating a scalable and usable workflow for use in notebooks.
    """
    
    ocr_patterns = {
    # Common OCR misreads 
    r'Fev\. ': 'Rev. ',
    r'Pev\. ': 'Rev. ',
    r'Bev\* ': 'Rev. ',
    r'Bev\. ': 'Rev. ',
    r'Pev\. ': 'Rev. ',
    r'Pev\* ': 'Rev. ',
    r'\^ev. ': 'Rev. ',
    r'Rev\* ': 'Rev. ',
    r'NE..BEBBY': 'NEWBERRY',
    r'Beaufoit': 'Beaufort',
    r'Dai lingt on': 'Darlington',
    r'Mr\* ': 'Mr. ',
    r', 3\* C\* ': ', S. C. ',
    r', 3\* C\. ': ', S. C. ',
    r', 3\. C ': ', S. C. ',
    r', 3\. 0\. ': ', S. C. ',
    r', 3\. G\. ': ', S. C. ',
    r'Mrs\* ': 'Mrs. ',
    r'CCUITY ': 'COUNTY ',
    r'Poute ': 'Route ',
    r'Pock Hill': 'Rock Hill',
    r'Bock Hill': 'Rock Hill',
    r'jegroes ': 'negroes ',
    r'febfuary ': 'february ',
    r'BROADW AY': 'BROADW AY',
    r'DAPLINGTOM ': 'DARLINGTON ',
    r'rogjstrstion ': 'registration ',
    r'Cher lesion ': 'Charleston ',
    r'travelling ': 'traveling ',
    r'crrolina ': 'carolina ',
    r'racisim ': 'racism ',
    r'supplios ': 'supplies ',
    r'mtcray ': 'mccray ',
    r'elininating': 'eliminating',
    r'mocray': 'mccray',
    r'John H\* ': 'John H. ',
    r'snartanburg': 'Spartanburg',
    r'tounded ': 'founded ',
    r'limcfow ': 'jimcrow ',
    r'hanvest ': 'harvest ',
    r'charpeston ': 'charleston ',
    r'Ifccray ': 'McCray ',
    r"colonialismimperialism ": "colonialism imperialism ",
    r"PRO GRESSIVE ": "PROGRESSIVE ",
    r"DEEOCRA TIC ": "DEMOCRATIC ",
    r"Preetent": "President",
    r"Harry S,": "Harry S.",
    r"Bespectfully": "Respectfully",
    r'Washington,D.C.': 'Washington, D.C.',
    r'S-TATE ': "STATE",
    r"foilowing": "following",
    }

    special_pattern = r'[^a-zA-Z0-9\s.,!?;:\'"()\-_/]{2,}'  # anything that's not letter, number, white space, or standard punctuation, in a sequence for 2 or more
    letter_pattern = r'\b\w*([a-zA-Z])\1{2,}\w*\b'          # sequences of 2+ of the same letter
    letter_pattern_4 = r'\b\w*([a-zA-Z])\1{4,}\w*\b'          # sequences of 4+ of the same letter

    # non_ascii -- any non-ascii character (should be 
    non_ascii = r'[^\x00-\x7F]'
    # non_ascii_between_ascii -- exclude whitespace around non-ascii (e.g. contentdm 72: securedΓÇönumber)
    non_ascii_between_ascii = r'(?<=\S)[^\x00-\x7F\s]+(?=\S)'

    general_exceptions = [
            r'^[a-zA-Z]{1,2}&[a-zA-Z]{1,2}$',           # e.g., a&b, AB&CD
            r'^[a-zA-Z]+=+[a-zA-Z]+$',                  # e.g., a=b, word=word
            r'^[a-zA-Z0-9]+\*[a-zA-Z0-9]+$',             # e.g., 2*3, 10*20 or didn*t
            # r"^[a-zA-Z]+\|-[a-zA-Z]+$"                   # e.g., word|-word 
        ]
    '''
    general_pattern example matches

    '''
    general_pattern = r'\b(?!(?:\w{1,2}&\w{1,2}|\w+=\w+|\d+\*\d+)$)\S*[@#$%^&*+=<>\|\`~[\]{}]\S*\b'

    def compile():
        """
        List comprehention to compile 'ocr_patterns' (dictionary) to be read 
        into regex standard functions

        Purpose:
            Regex needs the compiled the pattern regardless, and compiling them all ahead 
            of time stops our pattern from compiling every time we run the 're.sub' function
            (which we use in our 'clean_ocr_text()' function)
        Returns:
            A list of tuples in the form (compiled_pattern, replacement_string),
            ready to be used with standard regex functions.
        """
        return [(re.compile(p), r) for p, r in OCR_Clean.ocr_patterns.items()]
    
    def clean_ocr_text(text, compiled_patterns):
        """
        Clean the OCR errors from the text

        Parameters:
            text (string): entire transcript from digital collections
            compiled_patterns (list): list compiled via 'compile()' function
        Returns:
            text (string): cleaned transcript
        """
        for pattern, replacement in compiled_patterns:
            text = pattern.sub(replacement, text)
        return text

class OCR_Check:
    def is_messy_score(text):
        if not isinstance(text, str):
            return float('nan')
        # Count suspicious patterns
        suspicious_chars = len(re.findall(r'[\^&<>]', text))
        weird_spacing = len(re.findall(r'\s{2,}|\S{15,}', text))
        mixed_case = len(re.findall(r'[a-z][A-Z]|[A-Z]{3,}[a-z]', text))
        
        total_chars = len(text)
        if total_chars == 0:
            return 0
        
        messy_score = (suspicious_chars + weird_spacing + mixed_case) / total_chars
        return round(messy_score, 3)
    
    def is_messy_text(messy_score, threshold=0.3):
        return messy_score > threshold