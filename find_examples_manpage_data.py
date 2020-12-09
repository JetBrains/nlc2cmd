import argparse
import html.parser
import os
import re
import pandas as pd
from tqdm.auto import tqdm


class HTMLTextParser(html.parser.HTMLParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []

    def handle_data(self, data):
        self.data.append(data)


def check_query(query):
    return query[0].isupper()


def get_examples(command):
    name = command['name']
    examples = []

    regex_name = re.compile(fr"\b{re.escape(name)}\b")

    for p in command['paragraphs']:
        text = p['text'].strip()
        parser = HTMLTextParser()
        parser.feed(text)
        p['text'] = "".join(parser.data)

    query = ""

    for i, p in enumerate(command['paragraphs']):
        if p['section'] is None:
            continue
        section = p['section'].lower()
        if not "example" in section:
            continue
        example = p['text']

        found = False
        if example.startswith(name):
            found = True
        if not found and example.startswith("$ "):
            example = example[2:]
            found = True
        if not found:
            for match in regex_name.finditer(example):
                if match.start(0) > 10:
                    query = example[:match.start(0)].strip()
                example = example[match.start(0):]

                found = True
                break

        if not found:
            continue

        example = re.sub(r"\\\s+", "", example)
        if example.find('\n') != -1:
            last_index = example.find('\n')
            if check_query(example[last_index:].strip()):
                query = example[last_index:].strip()
            example = example[:last_index].strip()

        if i > 0:
            line = command['paragraphs'][i - 1]['text']
            if line.endswith(":"):
                query = line[:-1]
        if not query and i + 1 < len(command['paragraphs']):
            line = command['paragraphs'][i + 1]['text']
            if check_query(line):
                query = line
        examples.append([name, example, query])
        query = ""
    return examples


def main(args):
    if not os.path.isfile(args.input):
        raise ValueError(f"Can't find file '{args.input}'")
    chunk_size = args.chunk_size
    data = pd.read_json(args.input, lines=True, chunksize=chunk_size)

    TOTAL_LINES = 36668
    total_iterations = (TOTAL_LINES + chunk_size - 1) // chunk_size

    examples = []
    examples_with_query = 0
    with tqdm(data, total=total_iterations) as progress_bar:
        for chunk in progress_bar:
            for i, command in chunk.iterrows():
                new_examples = get_examples(command)

                for ex in new_examples:
                    if ex[2]:
                        examples_with_query += 1

                examples.extend(new_examples)
                progress_bar.set_postfix({"examples": len(examples), "examples with query": examples_with_query})

    examples = pd.DataFrame(examples, columns=["name", "command", "context"])
    examples.to_csv(args.output, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crawl examples from manpage-data.json")
    parser.add_argument("input", type=str, help="path to manpage-data.json")
    parser.add_argument("--chunk-size", type=int, default=100, help="number of lines in memory")
    parser.add_argument("-o", "--output", type=str, default="manpage-examples.csv", help="path to output file")
    args = parser.parse_args()
    main(args)
