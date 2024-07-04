import re
import random
from tqdm import tqdm

max_len = 512
setting = "none" # alternative: "all"


def shorten_pos_tag(tag):
    if tag == "NOUN":
        return "N"
    elif tag == "ADJ":
        return "A"
    elif tag == "VERB":
        return "V"
    elif tag == "ADV":
        return "R"
    elif tag == "PRON":
        return "P"
    elif tag == "DET":
        return "D"
    elif tag == "PART":
        return "T"
    elif tag == "CONJ":
        return "C"
    elif tag == "ADP":
        return "I"
    elif tag == "NUM":
        return "M"
    elif tag == "PUNCT":
        return "X"
    elif tag == "MANTRA":
        return "Y"
    else:
        return tag

def create_short_tag(morphosyntax):
    short_tag = ""
    if "=Nom" in morphosyntax:
        short_tag += "N"
    if "=Acc" in morphosyntax:
        short_tag += "A"
    if "=Dat" in morphosyntax:
        short_tag += "D"
    if "=Gen" in morphosyntax:
        short_tag += "G"
    if "=Voc" in morphosyntax:
        short_tag += "V"
    if "=Loc" in morphosyntax:
        short_tag += "L"
    if "=Ins" in morphosyntax:
        short_tag += "I"
    if "=Abl" in morphosyntax:
        short_tag += "B"
    if "=Fem" in morphosyntax:
        short_tag += "Fe"
    if "=Masc" in morphosyntax:
        short_tag += "Ma"
    if "=Neut" in morphosyntax:
        short_tag += "Ne"
     
    if "=Sing" in morphosyntax:
        short_tag += "S"
    if "=Plur" in morphosyntax:
        short_tag += "P"
    if "=Dual" in morphosyntax:
        short_tag += "Z"
    if "=Pres" in morphosyntax:
        short_tag += "Pr"
    if "=Past" in morphosyntax:
        short_tag += "Pa"
    if "=Fut" in morphosyntax:
        short_tag += "Fu"
    if "=Ind" in morphosyntax:
        short_tag += "In"
    if "=Imp" in morphosyntax:
        short_tag += "Im"
    if "=Cnd" in morphosyntax:
        short_tag += "Cn"
    if "=Opt" in morphosyntax:
        short_tag += "Op"
    if "=Part" in morphosyntax:
        short_tag += "Pt"

    return short_tag

def convert_punkt(string):
    if "comma" in string:
        return ","
    elif "fullStop" in string:
        return "."
    else:
        return ""


def turn_conllu_into_sentence_list(path, setting="all"):
    """
    This function reads a DCS conllu file and turns it into a list of serialized sentences for the ByT5 dependency parser.
    """
    sentences = []
    current_sentence_input = ""
    with open(path, "r") as f:
        for line in tqdm(f):
            if not "#" in line:
                tokens = line.split("\t")
                if len(tokens) > 1:                    
                    current_punct = ""
                    if "Punctuation" in tokens[9]:                        
                        current_punct = convert_punkt(tokens[9]) 
                    unsandhied = tokens[1]
                    pos = shorten_pos_tag(tokens[3])
                    feats = create_short_tag(tokens[5])                    
                    if "MANTRA" in pos:
                        pos = "MA"
                        feats = "MA"                    
                    if setting == "all":
                        current_sentence_input += unsandhied + "_" + pos + "_" + feats + current_punct + "_" + tokens[0] + " "
                    elif setting == "none":
                        current_sentence_input += unsandhied + "_" + tokens[0] + " "                    
                else:
                    sentences.append(current_sentence_input)
                    current_sentence_input = ""                    
    return sentences