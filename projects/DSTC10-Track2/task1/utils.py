#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DST utils."""

from collections import defaultdict
import copy
from itertools import chain
import logging
import json
import random
import re
import unicodedata

from fuzzywuzzy import fuzz


# DSTC10 Track-2 Task-1 DST schema
SCHEMA = {
    "attraction": {
        "book": [],
        "semi": ["type", "area", "name"]
    },
    "hotel": {
        "book": ["rooms", "stay", "day"],
        "semi": ["pricerange", "stars", "type", "area", "name"]
    },
    "restaurant": {
        "book": ["people", "time", "day"],
        "semi": ["food", "pricerange", "area", "name"]
    }
}
PUNC_CHARS = re.escape("!\"#$%&()*+,-./;<=>?@[\\]^_`{|}~")
PUNC_PATTERN = re.compile(f"[{PUNC_CHARS}]")


def get_logger(name, level=10):
    """Setup logger.

    Args:
        name (str): Logging name.
        level (int, optional): Logging level. Defaults to 10.

    Returns:
        logging.Logger: The logger after setting.
    """    
    logger = logging.getLogger(name)
    logger.propagate = 0
    logger.setLevel(level)
    header = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s %(filename)s[%(lineno)d]: %(message)s")
    header.setFormatter(formatter)
    logger.addHandler(header)
    return logger


logger = get_logger(__name__)


def flatten_ds(ds_dict, add_nm=True, remove_dc=False, multi_val_tag="<multi-val>"):
    """Flatten dialogue state from dict to sequence.

    Args:
        ds_dict (dict): The dialogue state dict.
        add_nm (bool, optional): If True, add `not mentioned`, `none` and empty slot. Defaults to True.
        remove_dc (bool, optional): If True, remove the `dontcare` slot value. Defaults to False.
        multi_val_tag (str, optional): The delimiter of multi slot values. Defaults to "<multi-val>".

    Returns:
        list: The sequence of dialogue state after flattening.
    """    
    ds_seq = []
    for dom in SCHEMA:
        for slot_type in SCHEMA[dom]:
            for slot in SCHEMA[dom][slot_type]:
                slot_tag = f"<{dom}-{slot_type}-{slot}>"
                vals = ds_dict.get(dom, {}).get(slot_type, {}).get(slot, ["not mentioned"])
                out_vals = []
                for val in vals:
                    if val in ["dont care", "don't care", "do n't care", "do nt care", "dontcare"]:
                        if remove_dc:
                            val = "<nm>"
                        else:
                            val = "<dc>"
                    elif val in ["not mentioned", "none", ""]:
                        val = "<nm>"
                    elif val in ["$today", "$tomorrow"]:
                        val = val[1:]
                    
                    if val == "<nm>" and not add_nm:
                        continue
                    out_vals.append(val)
                ds_seq.extend([slot_tag, f" {multi_val_tag} ".join(out_vals)])
    ds_seq = list(map(lambda x: x.lower(), ds_seq))
    return ds_seq


def parse_ds(ds_seq, remove_nm=True, convert_specials=True, multi_val_tag="<multi-val>", date_prefix=""):
    """Parse dialogue state from sequence to dict.

    Args:
        ds_seq (str): The sequence of dialogue state.
        remove_nm (bool, optional): If True, remove the `<nm>` slot value. Defaults to True.
        convert_specials (bool, optional): If True, convert special tokens `<nm>` and `<dc>` to
                                           `not mentioned` and `dontcare`. Defaults to True.
        multi_val_tag (str, optional): The delimiter of multi slot values. Defaults to "<multi-val>".
        date_prefix (str, optional): The prefix before date slot value. Defaults to "".

    Returns:
        dict: The dialogue state dict after parsing.
    """    
    ds_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    ds_seq = ds_seq.strip().split()
    cur_dom, cur_slot_type, cur_slot, cur_vals = "", "", "", []
    for token in ds_seq:
        if token in ["<ds>", "<ds/>", "</ds>", "</ds-"]:
            continue
        if token[0] == "<" and token[-1] == ">":
            if token.count("-") == 2:
                # dom-slot_type-slot special token
                # save previous slot and values
                if cur_dom and cur_slot_type and cur_slot and cur_vals:
                    ds_dict[cur_dom][cur_slot_type][cur_slot] = " ".join(cur_vals)
                cur_dom, cur_slot_type, cur_slot = token[1:-1].split("-")
                cur_vals = []
            elif token in (multi_val_tag, "<nm>", "<dc>"):
                cur_vals.append(token)
        else:
            cur_vals.append(token)

    if all([cur_dom, cur_slot_type, cur_slot, cur_vals]):
        ds_dict[cur_dom][cur_slot_type][cur_slot] = " ".join(cur_vals)

    out_ds_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for dom in ds_dict:
        for slot_type in ds_dict[dom]:
            for slot, slot_value in ds_dict[dom][slot_type].items():
                slot_values = list(map(lambda x: x.strip(), slot_value.split(multi_val_tag)))
                new_slot_values = []
                for slot_val in slot_values:
                    if slot_val.lower() in ("today", "tomorrow"):
                        slot_val = date_prefix + slot_val
                    if remove_nm and slot_val == "<nm>":
                        continue
                    if convert_specials:
                        if slot_val == "<nm>":
                            slot_val = "not mentioned"
                        elif slot_val == "<dc>":
                            slot_val = "dontcare"
                    new_slot_values.append(slot_val)
                if len(new_slot_values) > 0:
                    temp_slot_values = []
                    for v in new_slot_values:
                        if v not in temp_slot_values:
                            temp_slot_values.append(v)
                    out_ds_dict[dom][slot_type][slot] = temp_slot_values

    return out_ds_dict


def pre_process_utt(utt, role="user", num_to_eng=True, remove_punctuation=True):
    """Preprocess utterance.

    Args:
        utt (str): The utterance.
        role (str, optional): The role of utterance. Defaults to "user".
        num_to_eng (bool, optional): If True, convert arabic numerals to english. Defaults to True.
        remove_punctuation (bool, optional): If True, remove punctuation. Defaults to True.

    Returns:
        str: The utterance after preprocessing.
    """    
    utt = utt.strip().lower()
    # convert number to english in the utterance
    if num_to_eng:
        utt = number_to_english.run(utt, role=role)
    # remove punctuation
    if remove_punctuation:
        utt_ls = utt.split()
        for idx, token in enumerate(utt_ls):
            utt_ls[idx] = PUNC_PATTERN.sub(" ", token)
        utt = " ".join(utt_ls)
    # remove control chars
    utt = remove_control_chars(utt)
    # case by case replacing
    utt = replace_case(utt)
    # remove unnecessary white spaces
    utt = " ".join(utt.split())
    return utt


def remove_control_chars(utt):
    """Remove control characters in utterance.

    Args:
        utt (str): The utterance.

    Returns:
        str: The utterance after processing.
    """    
    utt = utt.strip().lower()
    if isinstance(utt, str):
        return "".join([(c if unicodedata.category(c)[0] != "C" else " ") for c in utt])
    elif isinstance(utt, list):
        return ["".join([(c if unicodedata.category(c)[0] != "C" else " ") for c in utt]) for u in utt]
    return utt


def replace_case(utt):
    """Replace some tokens in utterance case by case.

    Args:
        utt (str): The utterance.

    Returns:
        str: The utterance after processing.
    """    
    utt = f" {utt} "
    case_map = {
        " a. m. ": " am ",
        " p. m. ": " pm ",
        " a. m ": " am ",
        " p. m ": " pm ",
        " a m ": " am ",
        " p m ": " pm ",
        " a rm ": " am ",
        " p rm ": " pm ",
        " s. f. ": " sf ",
        " s. w. hotel ": " sw hotel ",
        "'": " '"
    }
    ordered_keys = sorted(case_map.keys(), key=lambda x: -len(x))
    for k in ordered_keys:
        utt = utt.replace(k, case_map[k])
    return utt


class NumberToEnglish(object):
    """Convert number to english in utterance."""
    def __init__(self):
        self.num_pat = re.compile(r"\d")
        self.time_pat = re.compile(r"(\d{1,2})(am|a\.m\.|pm|p\.m\.|o'clock)|(\d{1,2}:\d{2})"
                                   r"(am|a\.m\.|pm|p\.m\.|o'clock|)")
        self.time_suffix_pat = re.compile(r"(am|a\.m\.|a\.m|a m|a\. m|a\. m\.|pm|p\.m\.|p\.m|p m|p\. m|p\. m\.|"
                                          r"o'clock|o 'clock)( |\.|$)")
        self.train_pat = re.compile(r"tr\d{3,5}")
        self.star_range_pat = re.compile(r"\d-\d")
        self.decimals_pat = re.compile(r"(\d+?)\.(\d+)")
        self.num_constant = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                             7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
                             13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
                             18: "eighteen", 19: "nineteen"}
        self.in_hundred_constant = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty",
                                    7: "seventy", 8: "eighty", 9: "ninety"}

    def run(self, utt, role="user"):
        """Convert number to english in utterance.

        Args:
            utt (str): The utterance.
            role (str, optional): The role of utterance. Defaults to "user".

        Raises:
            NotImplementedError: Raise when meeting unknown `role`.

        Returns:
            str: The utterance after processing.
        """
        utt = utt.strip().lower()
        if role == "user":
            return self._run_user(utt)
        elif role == "sys":
            return self._run_sys(utt)
        else:
            raise NotImplementedError(f"unknown role: `{role}`")

    def _run_user(self, utt):
        """Convert number to english in user utterance.

        Args:
            utt (str): The user utterance.

        Returns:
            str: The user utterance after processing.
        """
        utt_ls = utt.split()
        for token_idx, token in enumerate(utt_ls):
            if self.num_pat.search(token):
                token_type = self._get_token_type(utt_ls, token_idx)
                if token_type == "time":
                    utt_ls[token_idx] = self._convert_time(utt_ls, token_idx)
                    continue
                elif token_type in ("phone", "postcode"):
                    utt_ls[token_idx] = self._convert_number(token)
                    continue
                # Otherwise, convert them to english directly
                token_ls = self._split_token(token)
                token_eng_ls = self._convert_numbers(token_ls)
                utt_ls[token_idx] = " ".join(token_eng_ls)

        utt = " ".join(utt_ls)
        utt = self._post_process(utt)
        return utt

    def _run_sys(self, utt):
        """Convert number to english in system utterance.

        Args:
            utt (str): The system utterance.

        Returns:
            str: The system utterance after processing.
        """
        utt_ls = utt.split()
        for token_idx, token in enumerate(utt_ls):
            if self.num_pat.search(token):
                token_type = self._get_token_type(utt_ls, token_idx)
                if token_type == "time":
                    utt_ls[token_idx] = self._convert_time(utt_ls, token_idx)
                    continue
                elif token_type == "star":
                    if self.star_range_pat.match(token):
                        utt_ls[token_idx] = self._convert_number(token[0]) + " to " + \
                            self._convert_number(token[2]) + f" {token[3:]}"
                        continue
                elif token_type == "price":
                    regex_res = self.decimals_pat.match(token)
                    if regex_res:
                        utt_ls[token_idx] = self._convert_number(regex_res.group(1)) + " point " + \
                            self._convert_number(regex_res.group(2)) + f" {token[regex_res.end():]}"
                        continue
                elif token_type in ("postcode", "reference"):
                    token_ls = self._split_token(token, alpha_alone=True)
                    token_eng_ls = self._convert_numbers(token_ls)
                    utt_ls[token_idx] = " ".join(token_eng_ls)
                    continue
                # otherwise, convert them to english directly
                token_ls = self._split_token(token)
                token_eng_ls = self._convert_numbers(token_ls)
                utt_ls[token_idx] = " ".join(token_eng_ls)

        utt = " ".join(utt_ls)
        utt = self._post_process(utt)
        return utt

    def _get_token_type(self, utt_ls, token_idx):
        """Get slot name for a specific token.

        Args:
            utt_ls (list): The utterance list.
            token_idx (int): The index of token.

        Returns:
            str: The slot name of this token.
        """
        # context size: 4
        token_context = utt_ls[max(0, token_idx - 4): token_idx + 5]
        token = utt_ls[token_idx]
        if self.time_pat.search(token):
            return "time"
        if self.train_pat.search(token):
            return "train"
        context_keywords = {
            "reference": ("reference", "ref", "ref#"),
            "phone": ("phone", "number"),
            "address": ("address", "street"),
            "star": ("star", "stars"),
            "postcode": ("postcode", "post", "code"),
            "price": ("pricerange", "price", "gbp", "pounds")
        }
        for token_type, keywords in context_keywords.items():
            for k in keywords:
                if k in token or k in token_context:
                    return token_type
        return "none"

    def _convert_time(self, utt_ls, token_idx):
        """Convert numeric representation of time to English representation.

        Args:
            utt_ls (list): The utterance list.
            token_idx (int): The index of time token.
        """
        def _24h_to_12h(time_hour, prob=0.2, default_suffix="", is_clock=True):
            out_time_suffix = default_suffix
            time_hour_int = int(time_hour)
            if time_hour_int > 12:
                # afternoon
                # convert 24-h to 12-h randomly
                if random.random() < prob:
                    time_hour = str(time_hour_int - 12)
                    out_time_suffix = "pm"
            if not out_time_suffix:
                # add a time suffix randomly
                if time_hour_int < 12:
                    suffix_choices = ["", "am"]
                elif time_hour_int > 12:
                    suffix_choices = ["", "pm"]
                else:
                    suffix_choices = [""]
                if is_clock:
                    suffix_choices.append("o'clock")
                out_time_suffix = random.choice(suffix_choices)

            time_hour_eng = self._convert_number(time_hour)
            return time_hour_eng, out_time_suffix

        token = utt_ls[token_idx]
        following_tokens = " ".join(utt_ls[token_idx + 1: token_idx + 3])
        regex_res = self.time_pat.search(token)
        time_suffix_regex_res = self.time_suffix_pat.match(following_tokens)
        if regex_res.group(1):
            # 8am
            time_hour = regex_res.group(1)
            time_hour_eng = self._convert_number(time_hour)
            time_suffix = regex_res.group(2)
            if time_suffix[0] == "p":
                time_hour_eng, time_suffix = _24h_to_12h(time_hour, default_suffix=time_suffix, is_clock=True)
            return f"{time_hour_eng} {time_suffix}".strip()
        elif regex_res.group(3):
            # 18:00pm / 9:30 a.m.
            time_hour, time_min = regex_res.group(3).split(":")
            time_hour_eng = self._convert_number(time_hour)
            time_suffix = regex_res.group(4) or (time_suffix_regex_res.group(1) if time_suffix_regex_res else "")
            if time_min == "00":
                if not time_suffix or time_suffix[0] == "p":
                    time_hour_eng, time_suffix = _24h_to_12h(time_hour, default_suffix=time_suffix, is_clock=True)
                if not time_suffix_regex_res:
                    return f"{time_hour_eng} {time_suffix}".strip()
                else:
                    return f"{time_hour_eng}"
            else:
                time_min_eng = self._convert_number(time_min)
                if not time_suffix or time_suffix[0] == "p":
                    time_hour_eng, time_suffix = _24h_to_12h(
                        time_hour, prob=0.1, default_suffix=time_suffix, is_clock=False
                    )
                if not time_suffix_regex_res:
                    return f"{time_hour_eng} {time_min_eng} {time_suffix}".strip()
                else:
                    return f"{time_hour_eng} {time_min_eng}"

    def _convert_numbers(self, tokens):
        """Convert arabic numerals to English representation.

        Args:
            tokens (list): The token list.

        Returns:
            list: The token list after processing.
        """
        return [self._convert_number(token) for token in tokens]

    def _convert_number(self, token):
        """Convert arabic numerals to English representation.

        Args:
            token (str): The token.

        Returns:
            str: The token after processing.
        """
        if not token.isdigit():
            return token
        if len(token) > 2:
            # single digit
            res = []
            for c in token:
                c_eng = self.num_constant.get(int(c), "")
                if c_eng:
                    res.append(c_eng)
            return " ".join(res)
        else:
            # whole number
            if int(token) in self.num_constant:
                return self.num_constant[int(token)]
            if token[1] == "0":
                return self.in_hundred_constant.get(int(token[0]), "")
            else:
                return self.in_hundred_constant.get(int(token[0]), "") + " " + self.num_constant.get(int(token[1]), "")

    def _split_token(self, token, alpha_alone=False):
        """Split English and arabic numerals.

        Args:
            token (str): The token.
            alpha_alone (bool, optional): If True, the letters inside the token are separated. Defaults to False.

        Returns:
            list: The token list after splitting.
        """
        token_ls = []
        num_str = eng_str = ""
        for c in token:
            if c.isdigit():
                if eng_str:
                    token_ls.append(eng_str)
                    eng_str = ""
                num_str += c
            else:
                if num_str:
                    token_ls.append(num_str)
                    num_str = ""
                if alpha_alone and eng_str:
                    token_ls.append(eng_str)
                    eng_str = ""
                eng_str += c
        if num_str:
            token_ls.append(num_str)
        if eng_str:
            token_ls.append(eng_str)
        return token_ls

    def _post_process(self, utt):
        """Post processing.

        Args:
            utt (str): The utterance.

        Returns:
            str: The utterance after post processing.
        """
        punctuation_map = {
            "@": "at",
            "&": "and",
            "#": "number",
            "%": "percent",
            "+": "",
            "-": "",
            "*": "",
            "<": "",
            ">": "",
            "`": "'"
        }
        utt_ls = utt.split()
        for idx, token in enumerate(utt_ls):
            for punc, punc_eng in punctuation_map.items():
                utt_ls[idx] = token.replace(punc, f" {punc_eng} ")
        utt = " ".join(utt_ls)
        return utt


def get_ds_delta(prev_ds_dict, cur_ds_dict):
    """Get the delta of dialogue state from previous turn to current turn.

    Args:
        prev_ds_dict (dict): The previous turn dialogue state.
        cur_ds_dict (dict): The current turn dialogue state.

    Returns:
        defaultdict: The delta of dialogue state.
    """
    ds_delta = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for dom, dom_ds in cur_ds_dict.items():
        for slot_type, slots in dom_ds.items():
            for slot, vals in slots.items():
                prev_vals = prev_ds_dict.get(dom, {}).get(slot_type, {}).get(slot, [])
                prev_vals = list(map(lambda x: x.lower(), prev_vals))
                vals_delta = []
                for val in vals:
                    if val.lower() not in prev_vals:
                        vals_delta.append(val)
                if len(vals_delta):
                    ds_delta[dom][slot_type][slot] = vals_delta
    return ds_delta


class PostProcess(object):
    """Postprocessing model generation sequence."""
    def __init__(self, db_path, normalization=True, db_guidance=True):
        self.db_engine = DBEngine(db_path)
        self.normalization = normalization
        self.db_guidance = db_guidance

    def run(self, ds_seq, prev_ds=None, utt_list=[]):
        """Run postprocessing.

        Args:
            ds_seq (str): The dialogue state of model generation.
            prev_ds (dict, optional): The previous dialogue state. Defaults to None.
            utt_list (list, optional): The dialogue history. Defaults to [].

        Returns:
            str: The dialogue state after postprocessing.
        """
        ds_seq = ds_seq.strip().lower()
        ds_dict = parse_ds(ds_seq)

        # case intervention
        if prev_ds is not None:
            prev_ds_dict = parse_ds(prev_ds, SCHEMA)
            ds_delta_dict = get_ds_delta(prev_ds_dict, ds_dict)
        else:
            ds_delta_dict = ds_dict
        self._case_intervention(ds_dict, ds_delta_dict, utt_list=utt_list)
        ds_dict = self._delete_redundant_slot_value(ds_dict)

        # filter slot value
        if prev_ds is not None:
            prev_ds_dict = parse_ds(prev_ds, SCHEMA)
            ds_delta_dict = get_ds_delta(prev_ds_dict, ds_dict)
        else:
            ds_delta_dict = ds_dict
        self._filter_slot_value(ds_dict, ds_delta_dict, utt_list=utt_list)
        ds_dict = self._delete_redundant_slot_value(ds_dict)

        # normalization
        db_match_score = {}
        if self.normalization:
            db_match_score = self._normalize(ds_dict, utt_list=utt_list)
            ds_dict = self._delete_redundant_slot_value(ds_dict)

        # DB guidance
        if self.db_guidance:
            self._db_guide(ds_dict, db_match_score, utt_list=utt_list)
            ds_dict = self._delete_redundant_slot_value(ds_dict)

        ds_seq = "<ds/> " + " ".join(flatten_ds(ds_dict)) + " </ds>"
        return ds_seq.lower()

    def _case_intervention(self, ds_dict, ds_delta_dict, utt_list=[]):
        """Run intervention case by case.

        Args:
            ds_dict (dict): The dialogue state dict.
            ds_delta_dict (dict): The delta of dialogue state.
            utt_list (list, optional): The dialogue history. Defaults to [].
        """
        slot_value_map = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        slot_value_map["hotel"]["semi"]["name"]["fairmont of san francisco"] = "fairmont san francisco"
        slot_value_map["attraction"]["semi"]["name"]["sf zoo"] = "san francisco zoo"

        # hotel-type-hotel
        hotel_type_no_hotel_prefix = (
            "looking for a a",
            "looking for a",
            "looking for",
            "finding a a",
            "finding a",
            "find a a",
            "finding",
            "there a",
            "find a",
            "have a",
            "find",
        )
        hotel_type_no_hotel_suffix = ()
        hotel_semi_type_vals = ds_delta_dict.get("hotel", {}).get("semi", {}).get("type", [])
        for hotel_semi_type_val in hotel_semi_type_vals:
            if hotel_semi_type_val.lower() == "hotel":
                # Get `hotel` index in the utterance
                hotel_indices = []
                target_utt_ls = []
                for utt in utt_list[::-1]:
                    target_utt_ls = utt.split()
                    for token_idx, token in enumerate(target_utt_ls):
                        if token.lower() == "hotel":
                            hotel_indices.append(token_idx)
                    if len(hotel_indices) > 0:
                        break
                # Judge whether `hotel` means `hotel-type`
                hotel_type_flags = []
                for token_idx in hotel_indices:
                    cur_hotel_type_flag = True  # Default: current hotel means `hotel-type`
                    token_prev_context = " ".join(target_utt_ls[max(token_idx - 4, 0): token_idx])
                    token_post_context = " ".join(target_utt_ls[token_idx + 1: token_idx + 3])
                    for _prefix in hotel_type_no_hotel_prefix:
                        if token_prev_context.endswith(_prefix):
                            cur_hotel_type_flag = False
                            break
                    else:
                        for _suffix in hotel_type_no_hotel_suffix:
                            if token_post_context.startswith(_suffix):
                                cur_hotel_type_flag = False
                                break
                    hotel_type_flags.append(cur_hotel_type_flag)
                if not all(hotel_type_flags):
                    # Wrong predictions
                    slot_value_map["hotel"]["semi"]["type"]["hotel"] = ""
                    logger.debug(f"Delete hotel-semi-type-hotel")

        # Replacement
        for dom, dom_ds in ds_dict.items():
            for slot_type, slots in dom_ds.items():
                for slot, slot_vals in slots.items():
                    for val_idx, slot_val in enumerate(slot_vals):
                        if slot_val.lower() in slot_value_map[dom][slot_type][slot]:
                            slot_vals[val_idx] = slot_value_map[dom][slot_type][slot][slot_val.lower()]

    def _filter_slot_value(self, ds_dict, ds_delta_dict, utt_list=[]):
        """Filter out `name` slot values that do not appear in the dialogue history.

        Args:
            ds_dict (dict): The dialogue state dict.
            ds_delta_dict (dict): The delta of dialogue state.
            utt_list (list, optional): The dialogue history. Defaults to [].
        """
        for dom, dom_ds in ds_delta_dict.items():
            dom_semi_ds = dom_ds.get("semi", {})
            for slot, vals in dom_semi_ds.items():
                # Filter `name` slot
                if slot == "name":
                    new_vals = []
                    for val in vals:
                        val_len = len(val.split())
                        val_match_flag = False
                        for utt in utt_list[::-1]:
                            target_utt_ls = utt.split()
                            for token_idx in range(0, len(target_utt_ls) - val_len + 1, val_len):
                                utt_tokens = " ".join(target_utt_ls[token_idx: token_idx + val_len])
                                match_score = (fuzz.partial_ratio(val.lower(), utt_tokens.lower()) +
                                               fuzz.ratio(val.lower(), utt_tokens.lower())) / 2
                                if match_score >= 40:
                                    new_vals.append(val)
                                    val_match_flag = True
                                    break
                            if val_match_flag:
                                break
                        else:
                            logger.debug(f"Delete mismatched slot value: {dom}-name: {val}")
                    if len(new_vals) > 0:
                        ds_dict[dom]["semi"][slot] = new_vals

    def _normalize(self, ds_dict, utt_list=[]):
        """Normalize the generated slot values according to the database.

        Args:
            ds_dict (dict): The dialogue state dict.
            utt_list (list, optional): The dialogue history. Defaults to [].

        Returns:
            defaultdict: Matching score of predicted values for each slot to the database.
        """
        db_match_score = defaultdict(dict)
        # process `semi` slot type
        for dom, dom_ds in ds_dict.items():
            dom_semi_ds = dom_ds.get("semi", {})
            # process multi slots
            remainder_slots = self._multi_slot_normalization(dom, dom_semi_ds, db_match_score)
            # process single slot
            self.single_slot_normalization(dom, dom_semi_ds, remainder_slots, db_match_score)
        # 某个 domain 有 name 没 area，其他 domain 的 area 和该 domain 的 name 可以对应，则填进去 area
        name_area_status = {}
        for dom, dom_ds in ds_dict.items():
            name_area_status[dom] = dict.fromkeys(["name", "area", "db_area"], "")
            name_area_status[dom]["name"] = dom_ds.get("semi", {}).get("name", [""])[0]
            name_area_status[dom]["area"] = dom_ds.get("semi", {}).get("area", [""])[0]
            if name_area_status[dom]["name"]:
                db_res = self.db_engine.query(dom, constraints=[["name", name_area_status[dom]["name"]]])
                if len(db_res) > 0:
                    name_area_status[dom]["db_area"] = db_res[0]["area"]
                else:
                    logger.debug(f"No DB results: {dom}-semi-name-{name_area_status[dom]['name']}")
        for dom, dom_name_area_status in name_area_status.items():
            if dom_name_area_status["name"] and not dom_name_area_status["area"]:
                for other_dom, other_dom_name_area_status in name_area_status.items():
                    if other_dom == dom:
                        continue
                    if other_dom_name_area_status["area"] and \
                        other_dom_name_area_status["area"] == dom_name_area_status["db_area"]:
                        ds_dict[dom]["semi"]["area"] = [dom_name_area_status["db_area"]]
                        logger.debug(f"Fill {other_dom}'s area `{other_dom_name_area_status['area']}` into {dom}")

        # Process `book` slot type
        for dom, dom_ds in ds_dict.items():
            dom_book_ds = dom_ds.get("book", {})
            for slot, vals in dom_book_ds.items():
                # normalize `time` slot
                if slot == "time":
                    new_vals = []
                    for val in vals:
                        norm_time = self._normalize_time(val)
                        if val != norm_time:
                            logger.debug(f"Normalize: {val} -> {norm_time}")
                        new_vals.append(norm_time)
                    dom_book_ds["time"] = new_vals
            # fill default `day`: `today`
            if dom == "hotel":
                if "day" not in dom_book_ds and "stay" in dom_book_ds and \
                    ("rooms" in dom_book_ds or "people" in dom_book_ds):
                    dom_book_ds["day"] = ds_dict.get("restaurant", {}).get("book", {}).get("day", ["today"])
                    logger.debug(f"Fill default {dom}-book-day: {dom_book_ds['day']}")
                if "day" in dom_book_ds and "rooms" not in dom_book_ds:
                    dom_book_ds["rooms"] = ["1"]
                    logger.debug(f"Fill default {dom}-book-rooms: {dom_book_ds['rooms']}")
            elif dom == "restaurant":
                if "day" not in dom_book_ds and "time" in dom_book_ds and "people" in dom_book_ds:
                    dom_book_ds["day"] = ds_dict.get("hotel", {}).get("book", {}).get("day", ["today"])
                    logger.debug(f"Fill default {dom}-book-day: {dom_book_ds['day']}")
                if "time" in dom_book_ds:
                    for idx, time_val in enumerate(dom_book_ds["time"]):
                        try:
                            time_val_hour, time_val_min = time_val.strip().split(":")
                            if int(time_val_hour) < 11:
                                dom_book_ds["time"][idx] = ":".join([str(int(time_val_hour) + 12), time_val_min])
                                logger.debug(f"Legalize {dom}-book-time: {dom_book_ds['time'][idx]}")
                        except:
                            continue

        return db_match_score

    def _multi_slot_normalization(self, dom, dom_semi_ds, db_match_score):
        """Normalize the generated slot values according to the results of querying the database using multiple slots.

        Args:
            dom (str): The current domain.
            dom_semi_ds (dict): The semi dialogue state.
            db_match_score (dict): Matching score of predicted values for each slot to the database.

        Returns:
            list: Remaining slot names that are not normalized.
        """
        if dom == "restaurant":
            multi_slots = ["name", "area", "food"]
        else:
            multi_slots = ["name", "area", "type"]
        remainder_slots = multi_slots

        # 确保这三个槽位值都为单值: 多值不执行 multi-slot normalization
        for slot in multi_slots:
            slot_vals = dom_semi_ds.get(slot, [])
            if len(slot_vals) > 1:
                return remainder_slots

        # All 3 slots
        if all([dom_semi_ds.get(s, [""])[0] not in ("", "dontcare") for s in multi_slots]):
            multi_constraints = [[s, dom_semi_ds[s][0]] for s in multi_slots]
            multi_slots_db_res = self.db_engine.query(dom, constraints=multi_constraints)
            if len(multi_slots_db_res) == 0:
                multi_slots_db_res = self.db_engine.query(
                    dom, constraints=multi_constraints[:1], soft_constraints=multi_constraints[1:], fuzzy_threshold=80
                )
            if len(multi_slots_db_res) > 0:
                for slot in multi_slots:
                    db_match_score[dom][slot] = multi_slots_db_res[0]["slot_match_score"][slot]
                    if dom_semi_ds[slot][0].lower() != multi_slots_db_res[0][slot].lower():
                        logger.debug(f"Normalize (multi-slots): {dom_semi_ds[slot][0]}"
                                     f" -> {multi_slots_db_res[0][slot]}")
                        dom_semi_ds[slot] = [multi_slots_db_res[0][slot].lower()]
                remainder_slots = []
                return remainder_slots

        # `name` and `area` slots
        if all([dom_semi_ds.get(s, [""])[0] not in ("", "dontcare") for s in multi_slots[:2]]):
            multi_constraints = [[s, dom_semi_ds[s][0]] for s in multi_slots[:2]]
            multi_slots_db_res = self.db_engine.query(dom, constraints=multi_constraints)
            if len(multi_slots_db_res) == 0:
                multi_slots_db_res = self.db_engine.query(
                    dom, constraints=multi_constraints[:1], soft_constraints=multi_constraints[1:], fuzzy_threshold=80
                )
            if len(multi_slots_db_res) > 0:
                for slot in multi_slots[:2]:
                    db_match_score[dom][slot] = multi_slots_db_res[0]["slot_match_score"][slot]
                    if dom_semi_ds[slot][0].lower() != multi_slots_db_res[0][slot].lower():
                        logger.debug(f"Normalize (multi-slots): {dom_semi_ds[slot][0]}"
                                     f" -> {multi_slots_db_res[0][slot]}")
                        dom_semi_ds[slot] = [multi_slots_db_res[0][slot].lower()]
                remainder_slots = multi_slots[2:]
        return remainder_slots

    def single_slot_normalization(self, dom, dom_semi_ds, target_slots, db_match_score):
        """Normalize the generated slot values according to the results of querying the database using single slot.

        Args:
            dom (str): The current domain.
            dom_semi_ds (dict): The semi dialogue state.
            target_slots (list): The slot names to be normalized.
            db_match_score (dict): Matching score of predicted values for each slot to the database.
        """
        for slot in target_slots:
            slot_vals = dom_semi_ds.get(slot, [])
            for val_idx, val in enumerate(slot_vals):
                if val in ("dont care", "don't care", "do n't care",
                           "do nt care", "dontcare", "not mentioned", "none", ""):
                    continue
                logger.debug(f"Single slot: {dom}-{slot}")
                if slot == "area":
                    search_dom = None
                else:
                    search_dom = dom
                db_res = self.db_engine.query(search_dom, constraints=[[slot, val]])
                if len(db_res) == 0:
                    db_res = self.db_engine.query(search_dom, soft_constraints=[[slot, val]])
                    db_res = self._rerank(db_res, dom, slot, val)
                if len(db_res) == 0:
                    for other_dom in SCHEMA:
                        if other_dom == dom or slot not in SCHEMA[other_dom]["semi"]:
                            continue
                        other_dom_ds_res = self.db_engine.query(other_dom, constraints=[[slot, val]])
                        if len(other_dom_ds_res) > 0:
                            db_res.extend(other_dom_ds_res)
                        else:
                            other_dom_ds_res = self.db_engine.query(other_dom, soft_constraints=[[slot, val]])
                            db_res.extend(other_dom_ds_res)
                    db_res.sort(key=lambda x: -x["match_score"])
                    db_res = self._rerank(db_res, dom, slot, val)
                if len(db_res) > 0:
                    db_match_score[dom][slot] = db_res[0]["slot_match_score"][slot]
                    if val.lower() != db_res[0][slot].lower():
                        logger.debug(f"Normalize: {val} -> {db_res[0][slot]}")
                        slot_vals[val_idx] = db_res[0][slot].lower()
                else:
                    logger.debug(f"Normalize: {val} -> none")
                    slot_vals[val_idx] = ""

    def _normalize_time(self, t):
        """Normalize the generated time string.

        Args:
            t (str): The time string.

        Returns:
            str: The time string after normalization.
        """
        t = t.replace("：", ":")
        if len(t) == 0:
            return ""
        if len(t) < 3:
            # 9
            # 18
            if t.isdigit():
                if int(t) < 12:
                    return f"0{t}:00"
                else:
                    return f"{t}:00"
            return ""
        elif len(t) <= 4 and t.isdigit():
            # 945
            # 1330
            t = f"{t[:-2]}:{t[-2:]}"
            if len(t) < 5:
                t = f"0{t}"
            return self._normalize_time(t)
        elif t.endswith("am"):
            # 4am
            t = t.rstrip("am").strip()
            t = f"{t}:00"
            if len(t) < 5:
                return f"0{t}"
            return t
        elif t.endswith("pm"):
            # 8pm
            t = t.rstrip("pm").strip()
            if t != "12":
                t = int(t) + 12
            return f"{t}:00"
        elif ":" in t:
            t_ls = t.split(":")
            minute = int(t_ls[1])
            if minute > 59:
                t = t_ls[0] + ":00"
            else:
                t = t_ls[0] + ":" + t_ls[1]
            return t
        return ""

    def _rerank(self, db_res, dom, slot, val):
        """Rerank the matching result based on the database matching score of the specified slot.

        Args:
            db_res (list): Database matching result.
            dom (str): The current domain.
            slot (_type_): The slot name used to rank.
            val (_type_): The slot value.

        Returns:
            list: The sorted database matching result.
        """
        sorted_db_res = []
        match_score = 0
        for db_entry in db_res:
            if db_entry["match_score"] < match_score:
                break
            sorted_db_res.append(db_entry)
            match_score = db_entry["match_score"]
        # Sort by slot value length
        sorted_db_res.sort(key=lambda x: len(x[slot]))
        return sorted_db_res

    def _db_guide(self, ds_dict, db_match_score, utt_list=[], fuzzy_threshold=60):
        """Database guidance.

        Args:
            ds_dict (dict): The dialogue state.
            db_match_score (dict): Matching score of predicted values for each slot to the database.
            utt_list (list, optional): The dialogue history. Defaults to [].
            fuzzy_threshold (int, optional): Threshold for fuzzy matching of slots. Defaults to 60.
        """
        guide_slots = ("area", "type", "food", "pricerange", "stars")
        for dom, dom_ds in ds_dict.items():
            dom_semi_ds = dom_ds.get("semi", {})
            # 确保 name 不是多值
            # if len(dom_semi_ds.get("name", [])) > 1:
            #     return

            for slot, slot_vals in dom_semi_ds.items():
                # Use `name` slot guide other slots
                if slot == "name" and db_match_score.get(dom, {}).get(slot, 100) > 80:
                    val = slot_vals[0]
                    db_res = self.db_engine.query(dom, constraints=[[slot, val]])
                    if len(db_res) == 0:
                        db_res = self.db_engine.query(
                            dom, soft_constraints=[[slot, val]], fuzzy_threshold=fuzzy_threshold
                        )
                        self._rerank(db_res, dom, slot, val)
                    if len(db_res) > 0:
                        if val.lower() != db_res[0][slot].lower():
                            logger.debug(f"DB guide (name): {dom}-{slot}-{val} -> {db_res[0][slot]}")
                        for guide_slot in guide_slots:
                            if guide_slot not in dom_semi_ds or len(dom_semi_ds[guide_slot]) > 1:
                                continue
                            if dom_semi_ds[guide_slot][0].lower() != str(db_res[0][guide_slot]).lower():
                                logger.debug(f"DB guide: {dom}-{guide_slot}-"
                                             f"{dom_semi_ds[guide_slot][0]} -> {str(db_res[0][guide_slot])}")
                                dom_semi_ds[guide_slot] = [str(db_res[0][guide_slot]).lower()]
                    else:
                        logger.debug(f"No DB results: {dom}-{slot}-{val}")

    def _delete_redundant_slot_value(self, ds_dict):
        """Delete redundant slot values.

        Args:
            ds_dict (dict): The dialogue state.

        Returns:
            dict: The new dialogue state.
        """
        new_ds_dict = defaultdict(lambda: defaultdict(dict))
        for dom, dom_ds in ds_dict.items():
            for slot_type, slots in dom_ds.items():
                for slot, slot_vals in slots.items():
                    new_slot_vals = list(filter(lambda x: x not in ["", "<nm>", "not mentioned"], slot_vals))
                    if len(new_slot_vals) > 0:
                        new_ds_dict[dom][slot_type][slot] = new_slot_vals
        return new_ds_dict


class DBEngine(object):
    """Database engine for querying."""
    def __init__(self, db_path):
        with open(db_path, "r") as fin:
            self.dbs = json.load(fin)
            self.global_dbs = []
            for _, entries in self.dbs.items():
                self.global_dbs.extend(entries)

    def query(self, domain=None, constraints=[], soft_constraints=[], fuzzy_threshold=60):
        """Query database for a given domain and constraints.

        Args:
            domain (str, optional): The current domain. Defaults to None.
            constraints (list, optional): The hard constraints. Defaults to [].
            soft_constraints (list, optional): The soft constraints. Defaults to [].
            fuzzy_threshold (int, optional): The fuzzying threshold. Defaults to 60.

        Returns:
            list: Database query results that meet the constraints.
        """
        if domain is None:
            db_candidates = self.global_dbs
        else:
            db_candidates = self.dbs[domain]
        found = []
        for record in db_candidates:
            match_score = 0
            slot_match_score = {}
            constraints_iterator = zip(constraints, [False] * len(constraints))
            soft_constraints_iterator = zip(soft_constraints, [True] * len(soft_constraints))
            for (slot, val), fuzzy_match in chain(constraints_iterator, soft_constraints_iterator):
                if val in ["", "not mentioned", "dont care", "don't care", "dontcare", "do n't care"]:
                    continue
                try:
                    record_keys = [k.lower() for k in record]
                    if slot.lower() not in record_keys:
                        continue
                    if not fuzzy_match:
                        if val.strip().lower() != record[slot].strip().lower():
                            break
                        match_score += 100
                        slot_match_score[slot] = 100
                    else:
                        fuzzy_score = (fuzz.partial_ratio(val.strip().lower(), record[slot].strip().lower()) +
                                       fuzz.ratio(val.strip().lower(), record[slot].strip().lower())) / 2
                        if fuzzy_score < fuzzy_threshold:
                            break
                        match_score += fuzzy_score
                        slot_match_score[slot] = fuzzy_score
                except:
                    continue
            else:
                res = copy.deepcopy(record)
                res["match_score"] = match_score
                res["slot_match_score"] = slot_match_score
                found.append(res)

        self._lower_value(found)
        found.sort(key=lambda x: -x["match_score"])
        return found

    def _lower_value(self, candidates):
        """Lower slot values for candidates.

        Args:
            candidates (list): Lower case candidates.
        """
        for candidate in candidates:
            for k, v in candidate.items():
                if isinstance(v, str):
                    candidate[k] = v.lower()


number_to_english = NumberToEnglish()
