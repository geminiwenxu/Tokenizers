import pickle
from os import PathLike
from pathlib import Path
from typing import Iterable

import streamlit as st
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer


class FlotaTokenizer:
    def __init__(self, vocab: Iterable[str], max_len=512, special=('##', '▁'), strict=False):
        super().__init__()
        self.vocab = set(vocab)
        self.special = special
        self.strict = strict
        self.max_len = max_len

    def tokenize(self, w, k):
        flota_dict = self.get_flota_dict(w, k)
        return [subword for i, subword in sorted(flota_dict.items())]

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


@st.experimental_memo(show_spinner=True)
def reload_flota(flota_model):
    vocab = get_vocab(flota_model)
    st.session_state.flota = FlotaTokenizer(
        vocab,
    )


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


if __name__ == '__main__':
    st.set_page_config(layout='wide')
    st.title("Tokenization Interactive Preview")
    st.subheader("FLOTA Tokenization")

    if 'flota_model' not in st.session_state:
        st.session_state.flota_model = "deepset/gbert-base"

    flota_model = st.text_input("Use Vocabulary from Model:", key='flota_model')
    reload_flota(flota_model)

    flota_k = st.number_input("k (Number of Iterations)", value=2, min_value=1)
    with st.form("input"):
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
            flota: FlotaTokenizer = st.session_state.flota
            st.code(
                "\n".join(
                    " ".join(flota.tokenize(w, flota_k)) for w in input_text.splitlines(False)
                ),
                "bash"
            )