# train.sh nl2bash.json manpage.json dev_dir cmd_options_help.csv 4 6
# augment by backtranslation
echo 'augmenting original data'
python augmentation/augment.py -c augmentation/configs/backtranslation_config.yml -i $1 -o $3/en-de-en-temp-sampling.json
python augmentation/backtranslate.py -i $1 -c 'en-de-en-ru-en' -o $3/en-de-en-ru-en.json -d cpu -t invocation -b 8
python augmentation/backtranslate.py -i $1 -c 'en-de-en' -o $3/en-de-en-ru-en.json -d cpu -t invocation -b 8
python augmentation/backtranslate.py -i $1 -c 'en-ru-en-de-en' -o $3/en-de-en-ru-en.json -d cpu -t invocation -b 8
python augmentation/backtranslate.py -i $1 -c 'en-ru-en' -o $3/en-de-en-ru-en.json -d cpu -t invocation -b 8

# generate new samples
echo 'generating new examples'
python generate_commands_from_synopsis.py $1 $2 --size 10 -o $3/generated.csv

# extract examples from manpage
echo 'extract examples from manpage'
python find_examples_manpage_data.py $2 -o dev_dir/manpage_examples.csv

# preprocessing
echo 'preprocessing'
python preprocessing.py $1 $3 $4

# train classifier
echo 'train classifier'
python model_clf.py $3 $3/clf_logdir -d cpu

# train context model
echo 'train ctx model'
python model_ctx.py $3 $3/ctx_logdir -d cpu

# create submission_code file
mv $3/*.vocab submission_code/
mv $3/*.model submission_code/
mv $3/clf_logdir/checkpoints/train.$5_full.pth submission_code/util_model.pth
mv $3/cmd_le submission_code/cmd_encoder
mv $3/ctx_logdir/checkpoints/train.$6_full.pth submission_code/ctx_model.pth

