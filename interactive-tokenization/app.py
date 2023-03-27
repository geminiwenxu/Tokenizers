import itertools
import pickle
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Iterable
from typing import Union, List, Tuple

import streamlit as st
from hunspell import HunSpell
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer

st.set_page_config(
    page_title="Home",
    layout='wide'
)
st.title("Tokenization Interactive Preview")

tab1, tab2 = st.tabs(["HunSpell", "FLOTA"])


def obey_tabs(text: str, tab_width: int = 4):
    new_text = []
    for line in text.splitlines(keepends=True):
        new_line = []
        for char in line:
            if char == '\t':
                new_line.extend((' ',) * (tab_width - (len(new_line) % tab_width)))
            else:
                new_line.append(char)
        new_text.append(''.join(new_line))
    return ''.join(new_text)


def identity(arg):
    return arg


def sub_to_next_closure(text, current_opening_level=2, closure="}"):
    substring = []
    for char in text:
        substring.append(char)
        if char == closure:
            current_opening_level -= 1
        if current_opening_level == 0:
            break
    return ''.join(substring)


def reload_hunspell():
    return HunSpell(
        'data/de_DE-morph.dic',
        'data/de_DE-morph.aff',
        verbose=True
    )


def parse_output(line: str, retain_tags: Tuple[str, ...], join_str: str, parenthesis_choice='longest'):
    if '(' in line and ')' in line:
        idx = line.find('(')
        closure = sub_to_next_closure(line[idx:], 1, ')')
        match parenthesis_choice:
            case 'longest':
                el = max(closure.strip('()').split('|'), key=lambda el: len(el.split()))
            case _:
                el = next(iter(closure.strip('()').split('|')))
        line = " ".join(map(str.strip, (line[:idx], el, line[idx + len(closure) + 1:])))

    retain_tags = set(retain_tags)
    return join_str.join([
        el.removeprefix(tag)
        for el, tag in itertools.product(line.split(), retain_tags)
        if el.startswith(tag)
    ])


def parse_analyze(line: str):
    return parse_output(line, ('ip:', 'st:', 'is:'), ' ')


def parse_stem(line: str):
    return parse_output(line, ('st:',), '')


def emtpy_to_na(results: List[Union[str, bytes]]):
    if results is None or len(results) == 0:
        return ["N/A"]
    elif type(results) is str:
        return results,
    else:
        return [
            result.decode() if type(result) is not str else result
            for result in results
        ]


def run(op, op_name, text, result_callback=None):
    if result_callback is None:
        result_callback = identity
    return [
        "\n".join([
            f'> {op_name}("{line.strip()}") = "{result_callback(result)}"'
            for result in
            emtpy_to_na(op(line.strip()))
        ]) for line in text.split()
    ]


def segment(text: str):
    results = run(st.session_state.hs.analyze, "segment", text, result_callback=parse_analyze)
    results = [
        "\n".join(list(dict.fromkeys(result.splitlines())))
        for result in results
    ]
    return "\n\n".join(results)


def analyze(text: str, do_parse=True):
    return "\n\n".join(
        run(st.session_state.hs.analyze, "analyze", text, result_callback=parse_analyze if do_parse else None))


def stem(text: str, do_parse=True):
    if do_parse:
        return "\n\n".join(run(st.session_state.hs.analyze, "stem", text, result_callback=parse_stem))
    else:
        return "\n\n".join(run(st.session_state.hs.stem, "stem", text))


def analyze_and_stem(text: str, do_parse=True):
    a_s = zip(
        run(st.session_state.hs.analyze, "analyze", text, result_callback=parse_analyze if do_parse else None),
        run(st.session_state.hs.analyze, "stem", text, result_callback=parse_stem if do_parse else None)
    )
    return "\n\n".join(["\n".join((a, s)) for a, s in a_s])


def suggest(text: str):
    return "\n\n".join(run(st.session_state.hs.suggest, "suggest", text))


def selected_analyze_or_stem(value):
    return value in ('Analyze & Stem', 'Analyze', 'Stem')


class FlotaTokenizer:
    def __init__(self, vocab: Iterable[str], max_len=512, special=('##', '▁'), strict=False):
        super().__init__()
        self.vocab = set(vocab)
        self.special = special
        self.strict = strict
        self.max_len = max_len

    def tokenize(self, w, k=2):
        flota_dict = self.get_flota_dict(w, k)
        return " ".join([subword for i, subword in sorted(flota_dict.items())])

    def get_flota_dict(self, w, k):
        max_subword, rest, i = self.max_subword_split(w)
        if max_subword is None:
            return {}
        if k == 1 or rest == len(rest) * '-':
            flota_dict = {i: max_subword}
            return flota_dict
        flota_dict = self.get_flota_dict(rest, k - 1)
        flota_dict[i] = max_subword
        return flota_dict

    def max_subword_split(self, w):
        for l in range(min(len(w), self.max_len), 0, -1):
            for i in range(0, len(w) - l + 1):
                if w[i] == '-':
                    continue
                subword = w[i:i + l]

                if subword in self.vocab:
                    return subword, w[:i] + l * '-' + w[i + l:], i
                elif not self.strict:
                    for special in self.special:
                        if special + subword in self.vocab:
                            return special + subword, w[:i] + l * '-' + w[i + l:], i
        return None, None, None


def autoload_vocab(path: str | PathLike):
    path = Path(path)

    match path.suffix:
        case '.p':
            with path.open('rb') as fp:
                vocab = pickle.load(fp)
        case '.txt':
            with path.open('r', encoding='utf-8') as fp:
                vocab = fp.read().splitlines(False)
        case _:
            raise ValueError()

    return vocab


@st.experimental_singleton(show_spinner=True)
def reload_flota(flota_model):
    vocab = get_vocab(flota_model)
    tokenizer = FlotaTokenizer(vocab, )
    st.session_state.flota = tokenizer
    return tokenizer


def get_vocab(flota_model):
    if (path := Path('data/').joinpath(flota_model)).exists():
        vocab = autoload_vocab(path)
    else:
        try:
            tokenizer: Tokenizer = Tokenizer.from_pretrained(flota_model)
            vocab = tokenizer.get_vocab(False)
        except Exception:
            try:
                tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
                    flota_model
                )
                vocab = tokenizer.get_vocab()
            except Exception as e:
                raise ValueError() from e
    return vocab


with tab1:
    st.subheader("HunSpell Morphological Segmentation")

    if 'hs' not in st.session_state:
        st.session_state.hs = reload_hunspell()

    mode = st.radio(
        "Mode",
        ['Segment', 'Analyze & Stem', 'Analyze', 'Stem', 'Suggest'],
        index=0, key="hs_mode",
    )
    do_parse = st.checkbox(
        'Parse Analyze Output',
        value=False, key="hs_parse",
        help="If toggled, will parse the output of `HunSpell.analyze` to create the output.",
        disabled=not selected_analyze_or_stem(mode)
    )

    with st.form("input_hunspell"):
        input_text = st.text_area(
            "Input Text",
            height=200,
            value="\n".join((
                "notlanden",
                "notgelandet",
                "gewetteifert",
                "wettgeeifert",
                "durchstudiert",
                "glattrasiert",
                "abzurechnender",
                "dunkle",
                "wärmste"
            )),
        )

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("Result:")
            if mode == "Segment":
                st.code(
                    segment(input_text)
                )
            elif mode == "Analyze & Stem":
                st.code(
                    analyze_and_stem(input_text, do_parse)
                )
            elif mode == "Analyze":
                st.code(
                    analyze(input_text, do_parse)
                )
            elif mode == "Stem":
                st.code(
                    stem(input_text, do_parse)
                )
            elif mode == "Suggest":
                st.code(
                    suggest(input_text)
                )

    expander = st.expander("Hunspell Dictionary & Affix Files", False)
    dic, aff = expander.tabs(["Dictionary", "Affixes"])

    with dic:
        st.code(
            obey_tabs(open("data/de_DE-morph.dic", 'r', encoding='utf-8').read()),
            'python'
        )

    with aff:
        st.code(
            obey_tabs(open("data/de_DE-morph.aff", 'r', encoding='utf-8').read()),
            'python'
        )

with tab2:
    st.subheader("FLOTA Tokenization")

    if 'flota_model' not in st.session_state:
        st.session_state.flota_model = "deepset/gbert-base"

    flota_model = st.text_input("Use Vocabulary from Model:", key='flota_model', placeholder="deepset/gbert-base",
                                help="For example: `deepset/gbert-base` or `xlm-roberta-base`")
    try:
        tokenizer = reload_flota(flota_model.strip())

        flota_k = st.number_input("k (Number of Iterations)", value=2, min_value=1)
        with st.form("input_flota"):
            input_text = st.text_area(
                "Input Text",
                height=200,
                value="\n".join((
                    "notlanden",
                    "notgelandet",
                    "gewetteifert",
                    "wettgeeifert",
                    "durchstudiert",
                    "glattrasiert",
                    "abzurechnender",
                    "dunkle",
                    "wärmste"
                )),
            )

            submitted = st.form_submit_button("Submit")
            if submitted:
                st.write("Result:")
                st.code(
                    "\n".join(run(
                        partial(tokenizer.tokenize, k=flota_k),
                        'flota',
                        input_text
                    ))
                )

    except Exception as e:
        print(e)
        st.error("Please choose a valid model")
