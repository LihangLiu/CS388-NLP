# java -cp /u/mooney/cs388-code/nlp/active-learning/stanford-corenlp-jar/*:. edu.stanford.nlp.parser.nndep.DependencyParser \
#   -trainFile wsj_00.conllx \
#   -testFile wsj_01.conllx \
#   -embedFile /projects/nlp/penn-dependencybank/en-cw.txt -embeddingSize 50 \
#   -model output_model \
#   -maxIter 50 \
#   -outFile annotations.conllx 

echo "random"
java -cp /u/mooney/cs388-code/nlp/active-learning/stanford-corenlp-jar/*:. DependencyParserActiveLearning 0 data/wsj_init.conllx data/wsj_01_03.conllx data/wsj_20.conllx
echo "sen_len"
java -cp /u/mooney/cs388-code/nlp/active-learning/stanford-corenlp-jar/*:. DependencyParserActiveLearning 1 data/wsj_init.conllx data/wsj_01_03.conllx data/wsj_20.conllx
echo "raw_prob"
java -cp /u/mooney/cs388-code/nlp/active-learning/stanford-corenlp-jar/*:. DependencyParserActiveLearning 2 data/wsj_init.conllx data/wsj_01_03.conllx data/wsj_20.conllx
echo "margin_prob"
java -cp /u/mooney/cs388-code/nlp/active-learning/stanford-corenlp-jar/*:. DependencyParserActiveLearning 3 data/wsj_init.conllx data/wsj_01_03.conllx data/wsj_20.conllx
