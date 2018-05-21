# CS 388 Natural Language Processing Homework 3: Active Learning for Neural Dependency Parsing
by Lihang Liu

## Directories:
data: contains the dataset.
stanford-corenlp-jar: nlp toolkits from stanford.
outputs: contains saved models and temporary files.
trace_files: trace files and LAS figures.

## How to Build

	./build.sh

## How to Run

	java -cp /u/mooney/cs388-code/nlp/active-learning/stanford-corenlp-jar/*:. DependencyParserActiveLearning [selection_function_id] [init_train_data_path] [unlabeled_train_pool_path] [test_data_path]

### Random Selction

	java -cp /u/mooney/cs388-code/nlp/active-learning/stanford-corenlp-jar/*:. DependencyParserActiveLearning 0 data/wsj_init.conllx data/wsj_01_03.conllx data/wsj_20.conllx

### Sentence Length

	java -cp /u/mooney/cs388-code/nlp/active-learning/stanford-corenlp-jar/*:. DependencyParserActiveLearning 1 data/wsj_init.conllx data/wsj_01_03.conllx data/wsj_20.conllx

### Raw Parse Probability

	java -cp /u/mooney/cs388-code/nlp/active-learning/stanford-corenlp-jar/*:. DependencyParserActiveLearning 2 data/wsj_init.conllx data/wsj_01_03.conllx data/wsj_20.conllx

### Margin Parse Probability

	java -cp /u/mooney/cs388-code/nlp/active-learning/stanford-corenlp-jar/*:. DependencyParserActiveLearning 3 data/wsj_init.conllx data/wsj_01_03.conllx data/wsj_20.conllx


