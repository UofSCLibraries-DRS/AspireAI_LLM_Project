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
        """
        Dictonary of messy OCR patterns and their clean counter part.
        Will serve simular to "STOP WORDS" in NLTK, with more flexiblity.

        Key: Messy OCR pattern (raw string)
        Value: Clean version of pattern (string)
        """
        r'_{2,}': ' ',     # two or more underscores = single white space
        r'—': '-',         # special long dash = standard dash
        r'\*{2,}': ' '     # two or more astricks (*) = single white space
    }

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