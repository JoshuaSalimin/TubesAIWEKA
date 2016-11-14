/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubesaiweka;


import java.io.*;
import java.util.*;
import weka.classifiers.*;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.clusterers.*;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 *
 * @author Joshua
 */
public class TubesAIWEKA {

   /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     */
    
    public static void save(Instances data) throws Exception {
     ArffSaver saver = new ArffSaver();
     saver.setInstances(data);
     saver.setFile(new File("C:\\Users\\Joshua\\Desktop\\TucilAI\\discretizediris.arff"));
     saver.writeBatch();
   }
    
    public static void save2(Instances data) throws Exception {
     ArffSaver saver = new ArffSaver();
     saver.setInstances(data);
     saver.setFile(new File("C:\\Users\\Joshua\\Desktop\\TucilAI\\output.arff"));
     saver.writeBatch();
   }
    
    public static Instances filter(Instances data) throws Exception{
        Discretize filter = new Discretize();
        filter.setInputFormat(data);
        Instances outputDataset;
        outputDataset = Filter.useFilter(data, filter);
        return outputDataset;
    }
    
    public static void main(String[] args) throws IOException, Exception {
        // TODO code application logic here
        //membaca dataset yang diberikan secara hard code
        DataSource reader = new DataSource("C:\\Users\\Joshua\\Desktop\\TucilAI\\iris.arff");
        Instances dataset = reader.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);
        Instances FilteredDataset = filter(dataset);
        Scanner in = new Scanner(System.in);
        int input = 1;
        Classifier classifier = new J48();
        classifier.buildClassifier(dataset);
        Classifier FilteredClassifier = new J48();
        FilteredClassifier.buildClassifier(FilteredDataset);
        
        while(input !=0){
            System.out.println("1. Discrete Filter");
            System.out.println("2. 10-fold Cross Validation");
            System.out.println("3. 10-fold Cross Validation dengan Discretize Filter");
            System.out.println("4. Full Training");
            System.out.println("5. Full Training dengan Discretize Filter");
            System.out.println("6. Save Model ke File Eksternal");
            System.out.println("7. Membaca Model dari File Eksternal");
            System.out.println("8. Membuat Instance Baru");
            System.out.println("9. Output Dataset");
            System.out.println("10. Test Instance");
            System.out.println("0. Keluar");
            input = in.nextInt();
            switch (input) {
                case 1:
                    System.out.println("discretizediris.arff Telah berhasil di simpan!");
                    Instances output = filter(dataset);
                    save(output);
                    break;
                case 2:
                    Evaluation evaluation = new Evaluation(dataset);
                    evaluation.crossValidateModel(classifier, dataset, 10, new Random(1));
                    System.out.println(evaluation.toSummaryString());
                    System.out.println(classifier);
                    break;
                case 3:
                    Evaluation evaluation2 = new Evaluation(FilteredDataset);
                    evaluation2.crossValidateModel(classifier, FilteredDataset, 10, new Random(1));
                    System.out.println(evaluation2.toSummaryString());
                    System.out.println(classifier);
                    break;
                case 4:
                    Evaluation evaluation3 = new Evaluation(dataset);
                    evaluation3.evaluateModel(classifier, dataset);
                    System.out.println(evaluation3.toSummaryString());
                    System.out.println(classifier);
                    break;
                 case 5:
                    Evaluation evaluation4 = new Evaluation(FilteredDataset);
                    evaluation4.evaluateModel(FilteredClassifier, FilteredDataset);
                    System.out.println(evaluation4.toSummaryString());
                    System.out.println(classifier);
                    break; 
                case 6:
                    weka.core.SerializationHelper.write("C:\\Users\\Joshua\\Desktop\\TucilAI\\output.model",classifier);
                    System.out.println("File Eksternal output.model telah berhasil disimpan");
                    System.out.println(" ");
                    break;
                case 7:
                    classifier = (Classifier) weka.core.SerializationHelper.read( new FileInputStream("C:\\Users\\Joshua\\Desktop\\TucilAI\\output.model"));
                    System.out.println("Model telah berhasil dibaca!");
                    System.out.println(" ");
                    System.out.println(classifier);
                    break;
                case 8:
                    int jumlah = dataset.numAttributes();
                    Instance instans = new DenseInstance(jumlah);
                    for(int i = 0; i < jumlah; i++){
                        System.out.println("attribute" + (i+1));
                        double x = in.nextDouble();
                        instans.setValue(i, x);
                    }   
                    dataset.add(instans);
                    System.out.println("Instance berhasil ditambahkan!");
                    break;
                case 9:
                    save2(dataset);
                    System.out.println("Dataset telah disimpan!");
                    break;
                case 10:
                    int jumlah2 = dataset.numAttributes();
                    Instance instans2 = new DenseInstance(jumlah2);
                    for(int i = 0; i < jumlah2-1; i++){
                        System.out.println("attribute" + (i+1));
                        double x = in.nextDouble();
                        instans2.setValue(i, x);
                    }   
                    instans2.setDataset(dataset);
                    double kelas = classifier.classifyInstance(instans2);
                    System.out.println("Hasil Prediksi : " + dataset.classAttribute().value( (int) kelas));
                    break;
                default:
                    System.out.println("Masukan salah, silahkan ulangi !");                  
                    break;
            }
        }
        System.out.println("Terima Kasih Telah Menggunakan Program Kami!");
        System.exit(0);
    }    
    
}
