import java.util.*;
import java.io.File;

public class DataWrapperClass {
	public int num_data;		// number of data (N)
	public int num_features;	// number of features (D)
	public int num_classes;		// number of different classes (K)
	public int num_cont_fea; 	// number of continuous features
	public int num_cat_fea;		// number of categorical features
	public ArrayList<ArrayList<Double>> continuous_features;	// only continuous features
	public ArrayList<ArrayList<Integer>> categorical_features;	// only categorical features
	public ArrayList<Integer> labels;	// labels of all data
	
	// read features and labels from input files
	public DataWrapperClass(String feature_fname, String label_fname){
        if(feature_fname.contains("CAT_")&&label_fname.contains("CAT_")){
            categorical_features = new ArrayList<>();
            categorical(feature_fname, label_fname);
            num_cat_fea = categorical_features.size();
            num_features = num_cat_fea;
        }else{
            continuous_features = new ArrayList<>();
            continuous(feature_fname, label_fname);
            num_cont_fea = continuous_features.size();
            num_features = num_cont_fea;
        }
	}
	
    private void categorical(String feature_fname, String label_fname){
        //scan features
        try{
        	File file = new File(feature_fname);
            Scanner feature = new Scanner(file);      
            Scanner currentLine = new Scanner(feature.nextLine());
            int col = 0, row = 1;
            while(currentLine.hasNext()){
            	currentLine.next();
            	col ++;
            }
            while(feature.hasNextLine()){
            	feature.nextLine();
            	row ++;
            }  
            num_data = row;
            feature.close();
            //re-scanner the file to get all data into a matrix array;
            feature = new Scanner(file);
            int data[][] = new int[row][col];
            for(int i=0; i<row; i++){
            	currentLine = new Scanner(feature.nextLine());
            	for(int j=0; j<col; j++){
            		if(currentLine.hasNext())
            			data[i][j] = currentLine.nextInt();
            	}
            }
            currentLine.close();
            feature.close();
            // reverse col and row
            for(int j=0; j<col; j++){
            	ArrayList<Integer> newList = new ArrayList<>();
            	for(int i=0; i<row; i++)
            		newList.add(data[i][j]);
            	categorical_features.add(newList);
            }
            
        }catch(Exception e){
            System.out.println("Categorical Feature: something is wrong");
        }
        // scan label
        labels = new ArrayList<>();
        try{
            Scanner label = new Scanner(new File(label_fname));
            
            while(label.hasNext()){
                Integer l = label.nextInt();
                if(!labels.contains(l)){
                	num_classes++;
                }
                labels.add(l);
            }  
            label.close();
            
        }catch(Exception e){
            System.out.println("Categorical Label: something is wrong");
        }
    }
    
    private void continuous(String feature_fname, String label_fname ){
        //scan features
        try{
        	File file = new File(feature_fname);
            Scanner feature = new Scanner(file);      
            Scanner currentLine = new Scanner(feature.nextLine());
            int col = 0, row = 1;
            while(currentLine.hasNext()){
            	currentLine.next();
            	col ++;
            }
            while(feature.hasNextLine()){
            	feature.nextLine();
            	row ++;
            }  
            num_data = row;
            feature.close();
            //re-scanner the file to get all data into a matrix array;
            feature = new Scanner(file);
            double data[][] = new double[row][col];
            for(int i=0; i<row; i++){
            	currentLine = new Scanner(feature.nextLine());
            	for(int j=0; j<col; j++){
            		if(currentLine.hasNext())
            			data[i][j] = currentLine.nextDouble();
            	}
            }
            currentLine.close();
            feature.close();
            // reverse col and row
            for(int j=0; j<col; j++){
            	ArrayList<Double> newList = new ArrayList<>();
            	for(int i=0; i<row; i++)
            		newList.add(data[i][j]);
            	continuous_features.add(newList);
            }
            
        }catch(Exception e){
            System.out.println("Continuous Feature: something is wrong");
        }
        // scan label
        labels = new ArrayList<>();
        try{
            Scanner label = new Scanner(new File(label_fname));
            
            while(label.hasNext()){
                Integer l = label.nextInt();
                if(!labels.contains(l)){
                	num_classes++;
                }
                labels.add(l);
            }  
            label.close();
            
        }catch(Exception e){
            System.out.println("Categorical Label: something is wrong");
        }
    }
    
	// static function, compare two label lists, report how many are correct
	public static int evaluate(ArrayList<Integer> l1, ArrayList<Integer> l2){
		int len = l1.size();
		assert len == l2.size();	// length should be equal
		assert len > 0;				// length should be bigger than zero
		int ct = 0;
		for(int i = 0; i < len; ++i){
			if(l1.get(i).equals(l2.get(i))) ++ct;
		}
		return ct;
	}
	
	public static double accuracy(ArrayList<Integer> l1, ArrayList<Integer> l2){
		int len = l1.size();
		assert len == l2.size();	// label lists should have equal length
		assert len > 0;				// lists should be non-empty
		double score = evaluate(l1,l2);
		score = score / len;		// normalize by divided by the length
		return score;
	}
}
