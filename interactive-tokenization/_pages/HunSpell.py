import itertools
from typing import Union, List, Tuple

import streamlit as st
from hunspell import HunSpell


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
        ]) for line in text.splitlines()
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


if __name__ == '__main__':
    st.set_page_config(layout='wide')
    st.title("Tokenization Interactive Preview")
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
