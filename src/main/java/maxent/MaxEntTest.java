package maxent;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import opennlp.tools.doccat.*;
import opennlp.tools.util.*;

/**
 * Created by Fahad Qureshi on 4/21/2018.
 */
public class MaxEntTest {
    DoccatModel model;

    public static void main(String[] args) {

        MaxEntTest twitterCategorizer = new MaxEntTest();
        twitterCategorizer.trainModel();
        twitterCategorizer.classifyNewTweet(new String[]{"Had a bad evening, need urgently a beer."});
    }

    public void trainModel() {

        try {

            File file = new File("input/tweets.txt");
            InputStreamFactory isf = new MarkableFileInputStreamFactory(file);
            ObjectStream<String> lineStream = new PlainTextByLineStream(isf , "UTF-8");
            ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);
            // Specifies the minimum number of times a feature must be seen'
            TrainingParameters tp = new TrainingParameters();
            tp.put(TrainingParameters.CUTOFF_PARAM, 2);
            tp.put(TrainingParameters.ITERATIONS_PARAM, 30);


            model = DocumentCategorizerME.train("en", sampleStream, tp, new DoccatFactory());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void classifyNewTweet(String[] tweet) {
        DocumentCategorizerME myCategorizer = new DocumentCategorizerME(model);
        double[] outcomes = myCategorizer.categorize(tweet);
        String category = myCategorizer.getBestCategory(outcomes);
        System.out.println(category);
       /* if (category.equalsIgnoreCase("1")) {
            System.out.println("The tweet is positive :) ");
        } else {
            System.out.println("The tweet is negative :( ");
        }*/
    }

}
















