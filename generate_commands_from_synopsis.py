import argparse
from functools import partial
import pandas as pd
from collections import Counter, defaultdict
import numpy as np
from tqdm.auto import tqdm
import warnings
import re

import sys
sys.path.append('../clai/utils')
import bashlint
from bashlint import data_tools


def update_graph(cmd, graph):
    parsed = data_tools.bash_parser(cmd)
    child = parsed.children[0]
    if not isinstance(child, bashlint.nast.PipelineNode):
        return
    prev_name = ""
    for c in child.children:
        if c.is_utility():
            cur_name = c.value
        else:
            cur_name = ""
        if prev_name and cur_name:
            graph[prev_name].add(cur_name)
        prev_name = cur_name

def add_utilities(cmd, counter):
    def get_utilities_fun(node):
        utilities = []
        if node.is_utility():
            utilities.append(node.value)
            for child in node.children:
                utilities.extend(get_utilities_fun(child))
        elif not node.is_argument():
            for child in node.children:
                utilities.extend(get_utilities_fun(child))
        return utilities

    parsed = data_tools.bash_parser(cmd)
    utils = get_utilities_fun(parsed)
    counter.update(utils)

def number_of_required_arguments(cmd):
    return sum([not x.optional for x in bashlint.grammar.bg.grammar[cmd].positional_arguments if isinstance(x, bashlint.grammar.ArgumentState)])


def get_options(name, manpage, alias_to_idx):
    change_name = {
        'grep': 'egrep',
        'gcc': 'aarch64-linux-gnu-gcc-8',
        'vim': 'rvim',
        'rename': 'file-rename'
    }

    name = change_name.get(name, name)

    x = manpage[manpage['name'] == name]
    if len(x) < 1:
        x = alias_to_idx.get(name, None)
        if x is None or len(x) < 1:
            return None
        else:
            x = manpage.loc[x]
    options = []
    paragraphs = x.iloc[0]['paragraphs']
    for p in paragraphs:
        text = p['text'].strip()
        if not p['is_option']:
            super_option_regex = r"^(?:\<[^>]+\>)?(-{1,2}\w+)(?:\<\/[^>]+\>)?[\t ]*"

            match = re.match(super_option_regex, text)
            if not match:
                continue
            p['short'] = []
            p['long'] = []
            p['expectsarg'] = None
            p['argument'] = None
            while match:
                found_text = match.group(1)
                if found_text.startswith("--"):
                    p['long'].append(found_text)
                else:
                    p['short'].append(found_text)

                text = re.sub(super_option_regex, "", text, 1)

                match = re.match(super_option_regex, text)
        if 'short' not in p:
            continue
        option = {
            'short': p['short'],
            'long': p['long'],
            'expectsarg': p['expectsarg'],
            'argument': p['argument']
        }
        if '\n' in text:
            text = text[text.find('\n') + 1:].strip()
        while text.startswith('<'):
            if '\n' in text:
                text = text[text.find('\n') + 1:].strip()
            else:
                text = re.sub("^\<[^>]+\>.*?\<\/[^>]+\>\s*", "", text).strip()
        option['text'] = text
        options.append(option)
    return options

def generate_single_command(results, avg_options=3, idx=None):
    if idx is None:
        idx = np.random.randint(len(results))
    row = results.iloc[idx]
    options = row['options']
    cmd = [row['cmd']]

    text = [row['synopsis']]

    if options:
        p = np.random.rand(len(options))
        if len(options) > avg_options:
            p = p < avg_options / len(options)
        else:
            p = p < 0.3
        p_options = [x for x, pp in zip(options, p) if pp]

        for x in p_options:
            if '--help' in x['long'] or '--version' in x['long'] or '-help' in x['short']:
                if len(p_options) > 1:
                    continue
            option_variants = x['short'] + x['long']
            option_var = np.random.choice(option_variants)
            current_option = [option_var]
            if x['expectsarg']:
                current_option.append('ARG')

            add_text = " ".join(x['text'].lower().split()[:5])
            text.append(add_text)
            cmd.append(tuple(current_option))
        cmd[1:] = sorted(cmd[1:], key=lambda x: "2"+x[0][2:] if x[0].startswith('--') else "1"+x[0][1:])
        real_cmd = [cmd[0]]
        for x in cmd[1:]:
            real_cmd.extend(x)
        cmd = real_cmd

    cmd.extend(["ARG"] * row['required'])
    return " ".join(cmd), " ".join(text)

def get_cmd_name(cmd):
    space_idx = cmd.find(' ')
    if space_idx == -1:
        return cmd
    return cmd[:space_idx]


def generate_commands(results, graph, avg_options=3, pipe_prob=0.3):
    idx = None
    cmds = []
    texts = []
    while True:
        cmd, text = generate_single_command(results, avg_options=avg_options, idx=idx)
        cmds.append(cmd)
        texts.append(text)
        p = np.random.rand()

        if p > pipe_prob:
            break

        cmd_name = get_cmd_name(cmd)
        next = graph.get(cmd_name, None)
        if next is None:
            break
        next_cmd = np.random.choice(list(graph[cmd_name]))
        idx = np.where(results['cmd'] == next_cmd)[0][0]

    cmd = " | ".join(cmds)
    text = " and ".join(texts)

    extra_cmds = ["{} | wc -l", "{} | grep ARG", "VAR=$({})", "VAR=`{}`"]
    extra_texts = [
        ["How many {}", "Count lines of {}", "Get number of {}"],
        ["Find all {} with ARG", "Which {} has ARG"],
        ["Set variable VAR to the {}", "Set variable VAR to {}"],
        ["Set variable VAR to the {}", "Set variable VAR to {}"]
    ]
    if np.random.rand() < 0.01:
        cmd_idx = np.random.randint(len(extra_cmds))
        cmd = extra_cmds[cmd_idx].format(cmd)
        text = np.random.choice(extra_texts[cmd_idx]).format(text)

    return cmd, text


def main(args):
    nl2bash = pd.read_json(args.nl2bash).T

    graph = defaultdict(lambda: set())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        tqdm.pandas(desc="Extracting utilities graph")
    nl2bash['cmd'].progress_apply(partial(update_graph, graph=graph))

    count_utilities = Counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        tqdm.pandas(desc="Extracting utilities from examples")
    nl2bash['cmd'].progress_apply(partial(add_utilities, counter=count_utilities))
    all_commands = list(bashlint.grammar.bg.grammar.keys())
    count_utilities.update(all_commands)

    commands = pd.DataFrame.from_dict(count_utilities, orient='index', columns=["count"]).reset_index() \
        .rename(columns={'index': 'cmd'}).sort_values('count').reset_index(drop=True)
    commands['required'] = commands['cmd'].apply(number_of_required_arguments)
    print(f"Found {len(commands)} total utilities")

    manpage = pd.read_json(args.manpage, lines=True)

    commands = commands.merge(manpage[['name', 'synopsis']], left_on='cmd', right_on='name', how='left')
    commands.loc[commands['synopsis'].isna(), 'synopsis'] = ''

    alias_to_idx = defaultdict(lambda: [])

    def get_aliases(x):
        idx = x.name
        for y in x['aliases']:
            y = y[0]
            alias_to_idx[y].append(idx)

    manpage.apply(get_aliases, axis=1)

    commands.drop_duplicates(inplace=True)
    print(f"Now {len(commands)} utilities")

    commands['options'] = commands['cmd'].apply(partial(get_options, manpage=manpage, alias_to_idx=alias_to_idx))
    del manpage
    results = []
    for t in tqdm(range(args.size), desc="Generating examples"):
        results.append(list(generate_commands(commands, graph)))
    results = pd.DataFrame(results, columns=["cmd", "query"])
    results.to_csv(args.output, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crawl examples from manpage-data.json")
    parser.add_argument("nl2bash", type=str, help="path to nl2bash-data.json")
    parser.add_argument("manpage", type=str, help="path to manpage-data.json")
    parser.add_argument("--size", type=int, default=10000, help="number of generated examples")
    parser.add_argument("-o", "--output", type=str, default="generated-examples.csv", help="path to output file")
    args = parser.parse_args()
    main(args)
