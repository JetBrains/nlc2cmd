import re
from submission_code.preprocessing import clean_text
import argparse
import pandas as pd
import numpy as np

import sys
sys.path.append('../clai/utils')
from bashlint.data_tools import bash_parser
from metric.metric_utils import compute_metric


def collect_data(nl2bash, dev_dir):
    train_data = []

    # original
    df = pd.read_json(nl2bash).T
    df['origin'] = 'original'
    train_data.append(df)

    # augmented
    df = pd.concat([pd.read_json(f'{dev_dir}/{name}', lines=True) for name in (
        # 'en-de-en-temp-sampling.json',
        'en-de-en-ru-en.json',
        'en-de-en.json',
        'en-ru-en-de-en.json',
        'en-ru-en.json',
    )])
    df = df[['invocation', 'cmd']].dropna()
    df['origin'] = 'augmented'
    train_data.append(df)

    # augmented
    df = pd.read_csv(f'{dev_dir}/generated.csv')
    df = df.rename(columns={'query': 'invocation'})
    df = df[['invocation', 'cmd']].dropna()
    df['origin'] = 'generated'
    train_data.append(df)

    # manpage examples
    df = pd.read_csv(f'{dev_dir}/manpage_examples.csv')
    df = df.rename(columns={'context': 'invocation', 'command': 'cmd'})
    df = df[['invocation', 'cmd']].dropna()
    df['origin'] = 'manpage'
    train_data.append(df)

    train_data = pd.concat(train_data)
    train_data['invocation'] = train_data['invocation'].apply(str.lower).apply(str.strip)
    train_data['cmd'] = train_data['cmd'].apply(str.strip)
    train_data = train_data.drop_duplicates().reset_index(drop=True)

    return train_data


def _clean_cmd(node):
    if node.kind.upper() == 'COMMANDSUBSTITUTION':
        r = ' $(' + node.value
        r += ' '.join([_clean_cmd(child) for child in node.children])
        r += ')'
        return r

    if node.kind.upper() == 'PROCESSSUBSTITUTION':
        r = ' ' + node.value + '('
        r += ' '.join([_clean_cmd(child) for child in node.children])
        r += ')'
        return r

    if node.kind.upper() == 'PIPELINE':
        r = '|'.join([_clean_cmd(child) for child in node.children])
        return r

    r = ' ' + node.value
    if node.kind.upper() == 'ARGUMENT':
        r = ' ARG'
    r += ' '.join([_clean_cmd(child) for child in node.children])
    if '::;' in node.value:
        r += ' \;'
    if '::+' in node.value:
        r += ' \+'

    return r


def clean_cmd(cmd):
    cmd = _clean_cmd(bash_parser(cmd)).replace('::;', '').replace('::+', '')
    cmd = cmd.strip()
    cmd = re.sub('\s+', ' ', cmd)
    return cmd


def remove_args(cmd):
    new_cmd = ''
    found = True

    while found:
        found = False
        parts = cmd.split('ARG')
        for i in range(1, len(parts)):
            new_cmd = 'ARG'.join(parts[:i]) + ' ' + 'ARG'.join(parts[i:])
            if compute_metric(new_cmd, 1, cmd) == 1:
                found |= True
                cmd = new_cmd
                break

    cmd = re.sub('\s+', ' ', cmd)
    cmd = cmd.strip()
    return cmd


def main(nl2bash, dev_dir, cmd_options):
    df = collect_data(nl2bash, dev_dir)

    df['cmd_cleaned'] = df['cmd'].apply(clean_cmd)
    df['text_cleaned'] = df['invocation'].apply(clean_text)
    df = df.drop_duplicates(subset=('cmd_cleaned', 'text_cleaned'))
    df = df.loc[df.cmd_cleaned.apply(lambda x: not x.startswith('root'))]
    df['cmd_cleaned'] = df['cmd_cleaned'].apply(remove_args)
    df = df.drop_duplicates(subset=('cmd_cleaned', 'text_cleaned'))
    df = df.loc[df.cmd_cleaned.apply(lambda x: not x.startswith('root'))]
    df.to_csv(f'{dev_dir}/train.csv')

    mandf = pd.read_csv(cmd_options)
    mandf = mandf.dropna(subset=['options']).reset_index(drop=True)

    def foo(options):
        options = eval(options)
        rlist = []
        for opt in options:
            rlist.append({
                'short': opt['short'],
                'long': opt['long'],
                'text': clean_text(opt['text']) if isinstance(opt['text'], str) else ' '.join([clean_text(x) for x in opt['text']])
            })
        return rlist

    mandf['cleaned_options'] = mandf.options.apply(foo)
    mandf.to_csv(f'{dev_dir}/man.csv')

    with open(f'{dev_dir}/text', 'w') as f:
        for x in df.text_cleaned:
            f.write(x.lower() + '\n')
        for x in mandf.cleaned_options:
            for opt in x:
                f.write(opt['text'].lower() + '\n')

    with open(f'{dev_dir}/cmd', 'w') as f:
        for x in df.cmd_cleaned:
            f.write(x + '\n')

    with open(f'{dev_dir}/all', 'w') as f:
        for x in df.text_cleaned:
            f.write(x.lower() + '\n')
        for x in mandf.cleaned_options:
            for opt in x:
                f.write(opt['text'].lower() + '\n')
        for x in df.cmd_cleaned:
            f.write(x + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('nl2bash', type=str)
    parser.add_argument("dev_dir", type=str)
    parser.add_argument("cmd_options", type=str)
    args = parser.parse_args()
    main(args.nl2bash, args.dev_dir, args.cmd_options)
