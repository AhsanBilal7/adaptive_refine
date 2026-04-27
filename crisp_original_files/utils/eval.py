import re
from math_verify import parse, verify

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)
    string = string.replace(",\\!", "")

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    
    string = string.replace("\\(", "")
    string = string.replace("\\)", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"
        
    if string.endswith(".00"):
        string = string[:-3]    
    

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string
 
def sort_expression(vec):
    sorted_vec = []
    for exp in vec:
        parts = exp.split('+')
        sorted_parts = sorted(parts)
        sorted_expression = '+'.join(sorted_parts)
        sorted_vec.append(sorted_expression)
    return sorted_vec

def norm_coef(vec):
    norm_vec = []
    for exp in vec:
        exp = re.sub(r'1([A-Z])', r'\1', exp)
        norm_vec.append(exp)
    return norm_vec
        
def match_rule(pred, label, kwargs):
    if not pred:
        return False 
    
    if kwargs['context'] == 'string':
        label = ''.join(label)
    else: 
        if kwargs['item']['op_type'] == 'add':
            pred = sort_expression(pred)
            label = sort_expression(label)
        elif kwargs['item']['op_type'] == 'map':
            pred = norm_coef(pred)
            label = norm_coef(label)
            
    match = (pred == label)
    return match

def match_fact(pred, label, kwargs):
    if not pred:
        return False 
    
    if kwargs['context'] == 'string':
        label = ''.join(label)
    elif kwargs['context'] == 'natural':
            pattern = r'([0-9]+)\b' 
            preds = []
            for obj in kwargs['item']['objects']:
                cnt = 0
                for s in pred:
                    if obj in s or obj.rstrip('s') in s:
                        matches = re.findall(pattern, s)
                        if matches:
                            cnt = int(matches[0])
                        break
                preds.append(cnt)
            pred = preds
                
    match = (pred == label)
    
    return match


def add_dollar_if_needed(s):
    # 检查字符串是否以$开始和结束
    s = s.strip()
    if not s.startswith('$'):
        s = '$' + s
    if not s.endswith('$'):
        s = s + '$'
    return s

def is_equiv(str1, str2, dataset):
    if dataset in ['gsm8k', 'math', 'olympiadbench', 'aime24', 's1k', 's1k-1.1','deepmath', 'openthoughts']:
        if str1 is None and str2 is None:
            print("WARNING: Both None")
            return True
        if str1 is None or str2 is None:
            return False
        # try:
        gold = parse(add_dollar_if_needed(str1))
        answer = parse(add_dollar_if_needed(str2))
        # Order here is important!
        return verify(gold, answer)
 
    else:
        if not str1 or not str2:
            return False
        return str1 and str2 and str1.lower() == str2.lower()
        

