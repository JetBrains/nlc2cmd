# NLC2CMD

This repository contains our contribution to [NLC2CMD Challenge](http://nlc2cmd.us-east.mybluemix.net/#/)

## Requirements

Install python dependencies:
```setup
pip install -r requirements.txt
```

Clone clai repository in the folder nearby
```
git clone https://github.com/IBM/clai.git
git checkout -t remotes/origin/nlc2cmd
```

Download preprocessed manpage data [cmd_options_help.csv](https://drive.google.com/file/d/19cWu7uNq0czo4g6jZuclflOzd5uLY--W/view?usp=sharing)

## Training

To train the model(s) in the paper, run this command:

```
./train.sh <path to nl2bash-data.json> <path to manpage-data.json> <path to dev dir> <cmd_options_help.csv> <best clf model epoch> <best ctx model epoch>
```
for example
```
mkdir dev_dir
./train.sh nl2bash-data.json manpage-data.json dev_dir cmd_options_help.csv 4 6
```

## Evaluation

To evaluate the model use tools from clai repository and submission_code folder.

## Pre-trained Models

You can download pretrained models here:
[pretrained models](https://drive.google.com/drive/folders/1KG3EiUe-dnqJg2v_yVBTZpE4TqU2iugM?usp=sharing) 

## Results

Our model achieves the following performance on :

### [NLC2CMD Challenge](http://nlc2cmd.us-east.mybluemix.net/#/)

| Model name         | Accuracy  | Energy (mW) |
| ------------------ |---------- | ----------- |
|        jb          |  0.499    |     828.9   |


## Contributing

All content in this repository is licensed under the MIT license.
