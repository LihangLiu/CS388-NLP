import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.parser.nndep.DependencyTree;
import edu.stanford.nlp.util.ScoredObject;
import edu.stanford.nlp.util.ScoredComparator;

import java.util.Properties;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.io.*;

public class DependencyParserActiveLearning {
	public static void main(String[] args) {
//		String trainPath = "data/wsj_init.conllx";
//		Dataset trainData = new Dataset(trainPath);
//		System.out.println(trainData.size());
//		
//		String outputPath = "outputs/tmp.conllx";
//		trainData.saved2file(outputPath);
		
		// arguments
		int maxEpoch = 20;
		int wordLimit = 1500;
		int evalMethodId = Integer.parseInt(args[0]);
		
		// load data
		String trainPath = args[1]; // "data/wsj_init.conllx";
		String unlabelPath = args[2]; // "data/wsj_01_03.conllx";
        String testPath = args[3]; // "data/wsj_20.conllx";
		Dataset trainData = new Dataset(trainPath);
		Dataset unlabelData = new Dataset(unlabelPath);
		Dataset testData = new Dataset(testPath);
		System.out.printf("Train:%d Unlabel:%d Test:%d \n", trainData.size(), unlabelData.size(), testData.size());
		
		// train
		EvalMethod evalMethod = EvalMethod.values()[evalMethodId];
		String modelPath = "outputs/tmp_model";
		String embeddingPath = "/projects/nlp/penn-dependencybank/en-cw.txt";
		ArrayList<Integer> list_num_train_words = new ArrayList<>();
		ArrayList<Double> list_LAS = new ArrayList<>();
		for (int i = 0; i < maxEpoch; i++) {
			System.out.printf("\n[Epoch %d]\n", i);
			// train
			System.out.println("\n==> Train");
			System.out.printf("Train size:%d\n", trainData.size());
			train(trainData, modelPath, embeddingPath);
			
			// test
			System.out.println("==> Test");
			System.out.printf("Test size:%d\n\n", testData.size());
			double las = test(testData, modelPath);
			System.out.println(las);
			
			// log
			list_num_train_words.add(trainData.num_words());
			list_LAS.add(las);
			
			// select new train data from unlabeled data
			System.out.println("==> Predict");
			System.out.printf("Predict size:%d\n\n", unlabelData.size());
			ArrayList<Double> predScores = predict(unlabelData, modelPath, evalMethod);
			sortListByList(predScores, unlabelData.entities);
			moveData(unlabelData, trainData, wordLimit);
		}
		System.out.println("==> Summary:");
		System.out.printf("Train:%d Unlabel:%d Test:%d \n", trainData.size(), unlabelData.size(), testData.size());
		System.out.println(evalMethod);
		System.out.printf("list_num_words_%d = ", evalMethodId);
		printIntegerList(list_num_train_words);
		System.out.printf("list_las_%d = ", evalMethodId);
		printDoubleList(list_LAS);
	}
	
	private static void train(Dataset trainData, String modelPath, String embeddingPath) {
		String tmpTrainPath = "outputs/tmp_train_data.conllx";
		trainData.saved2file(tmpTrainPath);
		
		Properties prop = new Properties();
        prop.setProperty("maxIter", "500");
        DependencyParser p = new DependencyParser(prop);
        p.train(tmpTrainPath, null, modelPath, embeddingPath);
	}
	
	private static double test(Dataset testData, String modelPath) {
		String testAnnotationsPath = "outputs/tmp_annotation_data.conllx";
		String tmpTestPath = "outputs/tmp_test_data.conllx";
		testData.saved2file(tmpTestPath);
		
		DependencyParser model = DependencyParser.loadFromModelFile(modelPath);
		double las = model.testCoNLL(tmpTestPath, testAnnotationsPath);
		return las;
	}
	
	private static ArrayList<Double> predict(Dataset predData, String modelPath, EvalMethod evalMethod) {
		ArrayList<Double> res = new ArrayList<>();
		
		String tmpPredPath = "outputs/tmp_pred_data.conllx";
		predData.saved2file(tmpPredPath);
        DependencyParser model = DependencyParser.loadFromModelFile(modelPath);
        List<DependencyTree> predictedParses = model.testCoNLLProb(tmpPredPath);
        
		switch (evalMethod) {
		case RANDOM:
			for (DataEntity cEntity : predData.entities) {
				res.add(Math.random());
			}
			break;
			
		case SEN_LEN:
			for (DataEntity cEntity : predData.entities) {
				double len = cEntity.size();
				res.add(-len);
			}
			break;
			
		case RAW_PROB:
			for (int i = 0; i < predData.size(); ++i) {
				DataEntity cEntity = predData.entities.get(i);
				DependencyTree tree = predictedParses.get(i);
				res.add(tree.RawScore / (2*cEntity.size()));
			}
			break;
			
		case MARGIN_PROB:
			for (int i = 0; i < predData.size(); ++i) {
				DataEntity cEntity = predData.entities.get(i);
				DependencyTree tree = predictedParses.get(i);
				res.add(tree.MarginScore / (2*cEntity.size()));
			}
			break;

		default:
			break;
		}
		
		return res;
	}
	
	private static void moveData(Dataset srcData, Dataset dstData, int wordLimit) {
		int wc = dstData.num_words();
		for (int i = srcData.size() - 1; i >= 0; i--) {
			DataEntity cEntity = srcData.entities.remove(i);
			dstData.entities.add(cEntity);
			if (dstData.num_words() - wc > wordLimit) {
				break;
			}
		}
	}
	
	// sort from big to small
	private static void sortListByList(ArrayList<Double> values, ArrayList<DataEntity> entities) {
		if (values.size() != entities.size()) {
			System.out.printf("SortList size don't match: %d, %d", values.size(), entities.size());
			System.exit(1);
		}
		int n = values.size();
		for (int i = 0; i < n; i++) {
			for (int j = 1; j < n - i; j++) {
				double cvalue = values.get(j);
				double pvalue = values.get(j-1);
				DataEntity ceneity = entities.get(j);
				DataEntity peneity = entities.get(j-1);
				if (pvalue < cvalue) {
					values.set(j-1, cvalue);
					values.set(j, pvalue);
					entities.set(j-1, ceneity);
					entities.set(j, peneity);
				}
			}			
		}
	}
	
	private static void printIntegerList(ArrayList<Integer> list) {
		System.out.print("[");
		for (Integer integer : list) {
			System.out.printf("%d, ", integer);
		}
		System.out.println("]");
	}
	
	private static void printDoubleList(ArrayList<Double> list) {
		System.out.print("[");
		for (Double d : list) {
			System.out.printf("%.4f, ", d);
		}
		System.out.println("]");
	}
	
    public static void main2(String[] args) {
        //  Training Data path
        // String trainPath = "data/wsj_init.conllx";
        String trainPath = "data/wsj_01_03.conllx";
        // Test Data Path
        String testPath = "data/wsj_20.conllx";
        // Path to embedding vectors file
        String embeddingPath = "/projects/nlp/penn-dependencybank/en-cw.txt";
        // Path where model is to be saved
        String modelPath = "outputs/model1";
        // Path where test data annotations are stored
        String testAnnotationsPath = "outputs/test_annotation.conllx";

        // Configuring propreties for the parser. A full list of properties can be found
        // here https://nlp.stanford.edu/software/nndep.shtml
        Properties prop = new Properties();
        prop.setProperty("maxIter", "20");
        DependencyParser p = new DependencyParser(prop);

        // Argument 1 - Training Path
        // Argument 2 - Dev Path (can be null)
        // Argument 3 - Path where model is saved
        // Argument 4 - Path to embedding vectors (can be null)
        p.train(trainPath, null, modelPath, embeddingPath);

        // Load a saved path
        DependencyParser model = DependencyParser.loadFromModelFile(modelPath);

        // Test model on test data, write annotations to testAnnotationsPath
        System.out.println(model.testCoNLL(testPath, testAnnotationsPath));

        // returns parse trees for all the sentences in test data using model, this function does not come with default parser and has been written for you
        List<DependencyTree> predictedParses = model.testCoNLLProb(testPath);

        // By default NN parser does not give you any probability 
        // https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf explains that the parsing is performed by picking the transition with the highest output in the final layer 
        // To get a certainty measure from the final layer output layer, we take use a softmax function.
        // For Raw Probability score We sum the logs of probability of every transition taken in the parse tree to get the following metric
        // For Margin Probability score we sum the log of margin between probabilities assigned to two top transitions at every step
        // Following line prints that probability metrics for 12-th sentence in test data
        // all probabilities in log space to reduce numerical errors. Adjust your code accordingly!
        System.out.printf("Raw Probability: %f\n",predictedParses.get(12).RawScore);
        System.out.printf("Margin Probability: %f\n",predictedParses.get(12).MarginScore);


        // You probably want to use the ScoredObject and scoredComparator classes for this assignment
        // https://nlp.stanford.edu/nlp/javadoc/javanlp-3.6.0/edu/stanford/nlp/util/ScoredObject.html
        // https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/util/ScoredComparator.html

    }
}

enum EvalMethod {
	RANDOM, SEN_LEN, RAW_PROB, MARGIN_PROB;
}

class Dataset {
	ArrayList<DataEntity> entities;
	
	public Dataset(String datapath) {
		entities = loadData(datapath);
	}
	
	public int size() {
		return entities.size();
	}
	
	public int num_words() {
		int res = 0;
		for (DataEntity cEntity : entities) {
			res += cEntity.size();
		}
		return res;
	}
	
	public void saved2file(String savedpath) {
		BufferedWriter writer = null;
        try {
            File file = new File(savedpath);

            writer = new BufferedWriter(new FileWriter(file));
            for (DataEntity cEntity : entities) {
				for (String line : cEntity.content) {
					writer.write(line + '\n');
				}
				writer.newLine();
			}
            System.out.println("Saved to:");
            System.out.println(file.getCanonicalPath());
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                writer.close();
            } catch (Exception e) {
            }
        }
	}
	
	private ArrayList<DataEntity> loadData(String datapath) {
		ArrayList<DataEntity> res = new ArrayList<>();
		
        try {
            FileReader fileReader = new FileReader(datapath);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            DataEntity cEntity = new DataEntity();
            String line = null;
            while((line = bufferedReader.readLine()) != null) {
//            		System.out.println(">>" + line + "<<");
//            		System.out.println(line.length());
                if (isNewLine(line)) {
                		res.add(cEntity);
                		cEntity = new DataEntity();
                		continue;
                }
                cEntity.addLine(line);
            }
            if (cEntity.content.size() > 0) {
            		res.add(cEntity);
			}
            
            bufferedReader.close();         
        }
        catch(FileNotFoundException ex) {
            System.out.println(
                "Unable to open file '" +  datapath + "'"
            );                
        }
        catch(IOException ex) {
            System.out.println(
                "Error reading file '" + datapath + "'"
            );                  
            // Or we could just do this: 
            // ex.printStackTrace();
        }
        return res;
	}
	
	private boolean isNewLine(String line) {
		if (line.length() == 0) {
			return true;
		}
		return false;
	}
}

class DataEntity {
	public ArrayList<String> content;
	
	public DataEntity() {
		content = new ArrayList<>();
	}
	
	public int size() {
		return content.size();
	}
	
	public void addLine(String line) {
		content.add(line);
	}
	
	public void print() {
		for (String line : content) {
			System.out.println(line);
		}
	}
}


