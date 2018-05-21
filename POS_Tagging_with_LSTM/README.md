# CS 388 Natural Language Processing Homework 2: Part-of-Speech Tagging with LSTMs
by Lihang Liu

## Directories:
src: contains the source code.
traces: contains the trace files and figures from tensorboard.
outputs: contains event files and checkpoints.

## How to Run

### Baseline Model
The original BiLSTMs model.

Train:

	cd src
	python pos_bilstm.py /projects/nlp/penn-treebank3/tagged/pos/wsj ../outputs/log-orth-baseline standard train none

Test:

	cd src
	python pos_bilstm.py /projects/nlp/penn-treebank3/tagged/pos/wsj ../outputs/log-orth-baseline standard test none	 

### Input Model
The BiLSTMs model with the orthographic features concatenated with the embedding layer.

Train:

	cd src
	python pos_bilstm.py /projects/nlp/penn-treebank3/tagged/pos/wsj ../outputs/log-orth-input standard train input

Test:

	cd src
	python pos_bilstm.py /projects/nlp/penn-treebank3/tagged/pos/wsj ../outputs/log-orth-input standard test input	 

### Output Model
The BiLSTMs model with the orthographic features concatenated with the LSTM output layer.

Train:

	cd src
	python pos_bilstm.py /projects/nlp/penn-treebank3/tagged/pos/wsj ../outputs/log-orth-output standard train output

Test:

	cd src
	python pos_bilstm.py /projects/nlp/penn-treebank3/tagged/pos/wsj ../outputs/log-orth-output standard test output	
