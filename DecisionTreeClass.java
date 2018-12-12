import java.util.*;

public class DecisionTreeClass {
	private class DecisionTreeNode implements Comparable<DecisionTreeNode>{
		public ArrayList<Integer> data_list; // list of data IDs
		public int opt_fea_type = -1;	// 0 if continuous, 1 if categorical
		public int opt_fea_id = -1;		// the index of the optimal feature
		public double opt_fea_thd = Double.NEGATIVE_INFINITY;	// the optimal splitting threshold 
																// for continuous feature
		public int opt_improvement = Integer.MIN_VALUE; // the improvement if split based on the optimal feature
		public boolean is_leaf = true;		// is it a leaf
		public int majority_class = -1;		// class prediction based on majority vote
		public int num_accurate = -1;		// number of accurate data using majority_class
		public DecisionTreeNode parent = null;		// parent node
		public ArrayList<DecisionTreeNode> children = null; 	// list of children when split
		
		public DecisionTreeNode(ArrayList<Integer> d_list, int m_class, int n_acc){
			data_list = new ArrayList<Integer>(d_list);
			majority_class = m_class;
			num_accurate = n_acc;
		}
		
		// change PriorityQueue order base on improvement
		public int compareTo(DecisionTreeNode node){
			if(this.equals(node)){
				return 0;
			}else if( this.opt_improvement > node.opt_improvement){
				return -1;
			}else
				return 1;
		}
	}
	
	public DataWrapperClass train_data;
	public int max_height;
	public int max_num_leaves;
	public int height;
	public int num_leaves;
	public DecisionTreeNode root;
	
	// constructor, build the decision tree using train_data, max_height and max_num_leaves
	public DecisionTreeClass(DataWrapperClass t_d, int m_h, int m_n_l){
		train_data = t_d;
		max_height = m_h;
		max_num_leaves = m_n_l;
		ArrayList<Integer> id = new ArrayList<>();
		for(int i = 0; i<train_data.labels.size(); i++){
			id.add(i);
		}
		root = new DecisionTreeNode(id, -1, -1);
		System.out.println("\n"+"Processing.....");
		setNode(root);
		
		// Tree
		PriorityQueue<DecisionTreeNode> maxHeap = new PriorityQueue<>();
		maxHeap.add(root);		
		System.out.println("Continue.....");
		
		while(!maxHeap.isEmpty()){
			DecisionTreeNode currentNode = maxHeap.poll();
			currentNode.children = new ArrayList<DecisionTreeNode>();
			if(currentNode.opt_improvement == 0){
				break;
			}else{
				if(currentNode.opt_fea_type == 1){
					// split data list of current node
					ArrayList<Integer>feature = train_data.categorical_features.get(currentNode.opt_fea_id);
					ArrayList<Integer> classType = findTypesIn(feature);
					ArrayList<ArrayList<Integer>> subLists = subIdLists_cat(feature, currentNode.data_list,classType);
					// give data sublist to child nodes
					for(ArrayList<Integer> subList :  subLists){
						currentNode.is_leaf = false;
						DecisionTreeNode child = new DecisionTreeNode(subList, -1, -1);
						currentNode.children.add(child);
						child.parent = currentNode;
						setNode(child);
						maxHeap.add(child);
					}
				}else{
					// split datalist by the threshold
					ArrayList<Double> feature = train_data.continuous_features.get(currentNode.opt_fea_id);
					double threshold = currentNode.opt_fea_thd;
					ArrayList<ArrayList<Integer>> subLists = subListWithThreshold_con(threshold, currentNode.data_list, feature);
					for(ArrayList<Integer> subList :  subLists){
						currentNode.is_leaf = false;
						DecisionTreeNode child = new DecisionTreeNode(subList, -1, -1);
						currentNode.children.add(child);
						child.parent = currentNode;
						setNode(child);
						//System.out.println("improve "+child.opt_improvement+" op feature " + child.opt_fea_id );
						maxHeap.add(child);
					}
				}
			}
			if(maxHeap.size() > max_num_leaves) break;
			if(getHeight(currentNode)> max_height) break;
			
		}
		System.out.println("Almost There.....");
	}

	public ArrayList<Integer> predict(DataWrapperClass test_data){
		ArrayList<Integer> preLabel = new ArrayList<>();
		int size_data = test_data.num_data;
		DecisionTreeNode node;
		DecisionTreeNode child;
		int i =0;
		while(i<size_data){
			node = root;
			while(node.children != null ){
				if(node.opt_fea_type == 1){
					// get values from features in test data
					int value = test_data.categorical_features.get(node.opt_fea_id).get(i);
					// choose which child node to go
					child = node.children.get(value);
					if(child.is_leaf){	
						preLabel.add(child.majority_class);
						break;
					}else{
						node = child;
					}
				}else{
					double threshold = node.opt_fea_thd;
					// get values from features in test data
					double value = test_data.continuous_features.get(node.opt_fea_id).get(i);
					// choose a child
					if(value< threshold)
						child = node.children.get(0);
					else
						child = node.children.get(1);
					
					if(child.is_leaf){
						preLabel.add(node.majority_class);
						break;
					}
					else
						node = child;
				}
			}
			i++;
		}
		System.out.println("Done.");
		return preLabel;
	}
	
	//General Function
	// re-do the computation for nodes like root
	private void setNode(DecisionTreeNode node){
		// find and set the majority class and the accuracy of node 
		setMajority(node);
		// find type classes in label
		ArrayList<Integer> typeClass = findTypesIn(train_data.labels);
		if(node.num_accurate == node.data_list.size()){
			node.opt_improvement = 0;
		}else{
			int bestAccur = Integer.MIN_VALUE;
			if(train_data.categorical_features !=null){
				int sum;
				for(ArrayList<Integer> feature: train_data.categorical_features){
					node.opt_fea_type = 1;
					// split sublists base on types of features
					ArrayList<Integer> typesInFeature = findTypesIn(feature);
					ArrayList<ArrayList<Integer>> idList = subIdLists_cat(feature, node.data_list,typesInFeature);
					// compute majority vote of different features
					sum = sumMajorityVote(idList, train_data.labels, typeClass);
					// get best votes between feature
					bestAccur = Math.max(bestAccur, sum);
					// get best feature to split
					if(bestAccur == sum){
						node.opt_fea_id = train_data.categorical_features.indexOf(feature);
					}
				}
			}else{
				node.opt_fea_type = 0;
				double bestThreshold;
				for(ArrayList<Double> feature: train_data.continuous_features){
					// sublist points feature to get new feature list
					ArrayList<Double> newFeature = newList(node.data_list, feature);
					//then sorting feature and find out threshold 
					ArrayList<Double> sortFeature = new ArrayList<>(newFeature);
					sort(sortFeature);
					int majorityVote = 0;
					int bestVote = Integer.MIN_VALUE;
					bestThreshold = Double.MIN_VALUE;
					for(int i = 0; i<sortFeature.size()-1; i++){
						// after sort, check every threshold and find the best
						ArrayList<ArrayList<Integer>> idList = idListForLabel(sortFeature, feature, node.data_list,i);
						majorityVote = sumMajorityVote(idList, train_data.labels, typeClass);
						// get the best vote over all thresholds in one feature 
						bestVote = Math.max(bestVote, majorityVote);
						if(bestVote == majorityVote){
							// set the best threshold in the feature
							bestThreshold = sortFeature.get(i)+0.000001;
						}
						//System.out.println("best vote "+ bestVote);
					}
					// get the best Accuracy over all feature
					bestAccur = Math.max(bestAccur, bestVote);
					if(bestAccur == bestVote){
						node.opt_fea_id = train_data.continuous_features.indexOf(feature);
						node.opt_fea_thd = bestThreshold;
					}
				}
			}
			node.opt_improvement = bestAccur - node.num_accurate;
		}
	}
	// check the height of tree
	private int getHeight(DecisionTreeNode node){
		DecisionTreeNode current = node;
		int count = 0;
		while(current.parent != null){
			count++;
			current = current.parent;
		}
		return count;
	}
	// get the types/classes of label or any categorical feature 
	private ArrayList<Integer> findTypesIn(ArrayList<Integer> list){
		ArrayList<Integer> type = new ArrayList<>();
		for(int element: list){
			if(!type.contains(element)){
				type.add(element);
			}
		}
		// sorting type, so sublist will generalize in increasing order
		Collections.sort(type);
		return type;
	}
	// set the majority class and the accuracy from label
	private void setMajority(DecisionTreeNode node){
		//System.out.println("new");
		ArrayList<Integer> listInlabel = new ArrayList<>();
		for(Integer i : node.data_list){
			listInlabel.add(train_data.labels.get(i));
		}
		ArrayList<Integer> typeClass = findTypesIn(listInlabel);
		int count, maxClass = Integer.MIN_VALUE;
		for(Integer type : typeClass){
			count = 0;
			for(Integer e : listInlabel){
				if(e == type){
					count++;
				}
			}
			maxClass = Math.max(maxClass, count);
			if(maxClass == count){
				node.majority_class = type;
				node.num_accurate = count;
			}
		}
	}
	//compute majority vote after feature split
	private int sumMajorityVote(ArrayList<ArrayList<Integer>> idLists, ArrayList<Integer> label, ArrayList<Integer> classType){
		int sum = 0;
		int count;
		int max;
		for(ArrayList<Integer> list: idLists){
			max = Integer.MIN_VALUE;
			for(int type : classType){
				count = 0;
				for(int id : list){
					if(label.get(id)==type) count++;
				}
				max = Math.max(max, count);
			}
			sum += max;
		}
		return sum;
	}	
	
	//continuous feature
	//get id Lists after split by a threshold
	private ArrayList<ArrayList<Integer>> idListForLabel(ArrayList<Double> sortedFeature, ArrayList<Double> feature, ArrayList<Integer> dataList,int pThr){
		ArrayList<ArrayList<Double>> subList = new ArrayList<>();
		ArrayList<Double> dlist1 = new ArrayList<>();
		ArrayList<Double> dlist2 = new ArrayList<>();
		double breakPoint = sortedFeature.get(pThr)+0.000001;
		// split sorted feature to two sublists by threshold
		for(double i : sortedFeature){
			if(i<breakPoint){
				dlist1.add(i);
			}else{
				dlist2.add(i);
			}
		}
		subList.add(dlist1);
		subList.add(dlist2);
		ArrayList<ArrayList<Integer>> idLists = new ArrayList<>();
		for(ArrayList<Double> list : subList){
			int i=0, j=0;
			ArrayList<Integer> idList = new ArrayList<>();
			while(i<list.size()){
				// take one value from sorted sublist and check the value through original feature(not the new feature) by using data List
				// feature.get(dataList.id) == value in sorted lists
				if(list.get(i).equals(feature.get(dataList.get(j)))){
					idList.add(dataList.get(j));
					i++;
				}
				if(j == dataList.size()-1){
					j=-1;
				}
				j++;
			}
			idLists.add(idList);
		}
		return idLists;
	}
	// sort for continuous feature
	private void sort(ArrayList<Double> feature){
		PriorityQueue<Double> temp = new PriorityQueue<>();
		for(Double value : feature){
			temp.add(value);
		}
		feature.clear();
		while(!temp.isEmpty()){
			feature.add(temp.poll());
		}
	}
	
	// get values from feature by using id
	private ArrayList<Double> newList(ArrayList<Integer> idList, ArrayList<Double> feature){
		ArrayList<Double> newList = new ArrayList<>();
		for(int i : idList){
			newList.add(feature.get(i));
		}
		return newList;
	}
	//split to sublists by a known threshold
	private ArrayList<ArrayList<Integer>> subListWithThreshold_con(double pThre, ArrayList<Integer> dataList, ArrayList<Double> unsortFeature){
		ArrayList<ArrayList<Integer>> idList = new ArrayList<>();
		ArrayList<Integer> list1 = new ArrayList<>();
		ArrayList<Integer> list2 = new ArrayList<>();
		for(int i=0; i < dataList.size(); i++){
			if(unsortFeature.get(i)<=pThre){
				list1.add(i);
			}else{
				list2.add(i);
			}
		}
		idList.add(list1);
		idList.add(list2);
		return idList;
	}
	
	//cat feature
	// split sublist by using types of feature
	private ArrayList<ArrayList<Integer>> subIdLists_cat(ArrayList<Integer> feature, ArrayList<Integer> dataList, ArrayList<Integer> typesInFeature){
		ArrayList<ArrayList<Integer>> idLists = new ArrayList<>();
		for(int type : typesInFeature){
			ArrayList<Integer> idList = new ArrayList<>();
			for(int id : dataList ){
				if(feature.get(id) == type){
					idList.add(id);
				}
			}
			idLists.add(idList);
		}
		return idLists;
	}
	
}
