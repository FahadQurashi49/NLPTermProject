package maxent;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizer;
import opennlp.tools.doccat.DocumentCategorizerME;

import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;

/**
 * oepnnlp version 1.7.2
 * Training of Document Categorizer using Maximum Entropy Model in OpenNLP
 * @author www.tutorialkart.com
 */
public class MaxEntTest2 {

    public static void main(String[] args) {

        try {
            // read the training data
            InputStreamFactory dataIn = new MarkableFileInputStreamFactory(new File("input/newdata.txt"));
            ObjectStream lineStream = new PlainTextByLineStream(dataIn, "UTF-8");
            ObjectStream sampleStream = new DocumentSampleStream(lineStream);

            // define the training parameters
            TrainingParameters params = new TrainingParameters();
            params.put(TrainingParameters.ITERATIONS_PARAM, 10+"");
            params.put(TrainingParameters.CUTOFF_PARAM, 0+"");

            // create a model from traning data
            DoccatModel model = DocumentCategorizerME.train("en", sampleStream, params, new DoccatFactory());
            System.out.println("\nModel is successfully trained.");

            // save the model to local
            BufferedOutputStream modelOut = new BufferedOutputStream(new FileOutputStream("model"+File.separator+"en-so-classifier-maxent.bin"));
            model.serialize(modelOut);
            System.out.println("\nTrained Model is saved locally at : "+"model"+File.separator+"en-so-classifier-maxent.bin");

            // test the model file by subjecting it to prediction
            DocumentCategorizer doccat = new DocumentCategorizerME(model);
            String[] docWords = "php n elements in if statement with or  this is what i want to do:   <pre><code>$myarray = [ value1    value2    value3  ...  valuen ];  if ($myarray[0] ==  string  || $myarray[1] ==  string  || $myarray[2] ==  string  ||...|| $myarray[n] ==  string ){  echo  error ;  } else {  echo  success ;  } </code></pre>   how can i solve it     thanks".replaceAll("[^A-Za-z]", " ").split(" ");
            double[] aProbs = doccat.categorize(docWords);

            // print the probabilities of the categories
            System.out.println("\n---------------------------------\nCategory : Probability\n---------------------------------");
            for(int i=0;i<doccat.getNumberOfCategories();i++){
                System.out.println(doccat.getCategory(i)+" : "+aProbs[i]);
            }
            System.out.println("---------------------------------");

            System.out.println("\n"+doccat.getBestCategory(aProbs)+" : is the predicted category for the given sentence.");
        }
        catch (IOException e) {
            System.out.println("An exception in reading the training file. Please check.");
            e.printStackTrace();
        }
    }
}