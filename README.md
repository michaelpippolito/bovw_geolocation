# Finding geolocation with BOVW

Three python codes:
  1. gen_dataset.py
  2. gen_vocab.py
  3. search.py

1. gen_dataset.py  
  Options:  
      -d, --dataset   	Directory to store the dataset  
      -r, --route     	Path to the Garmin GPX file  
      -t, --test      	True if and only if downloading images for tests  
      -q, --quiet     	True to prevent output  

2. gen_vocab.py  
  Options:  
      -d, --dataset   	Path to the dataset images  
      -k, --kmeans    	The value of k to cluster with  
      -c, --codebook  	Directory to store the codebooks  
      -q, --quiet     	True to prevent output  

3. search.py  
  Options:  
      -q, --query     	Path to the image or directory of images to be queried  
      -c, --codebook  	Path to the codebook  
      -l, --limit     	Number of images to return per search  
      --quiet         	True to prevent output  

# Instructions

1. Run gen_dataset.py
2. Run gen_vocab.py
3. Run search.py

# Requirements
Install OpenCV with contrib  
Install SciPy  
Install Cassandra  
