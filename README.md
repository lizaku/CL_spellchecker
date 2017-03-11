# The Spellchecker project for the CL course

## Architecture principles
* taking each word in text and computing all its possible variants that result from the following operations:
        * vowel substituion
        * symbols dubbing
        * reduction
        * insertion
        * combination of several operations
* selecting the best variant by computing the Hamming distance between variants and the original word
* or simply taking the most frequent word as found in the supplied corpus of texts
* PROFIT!!!
