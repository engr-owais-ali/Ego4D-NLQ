# Ego4D NLQ

Requirements: Required dataset folder (universal for all notebooks) (https://drive.google.com/drive/folders/1a9Tfdi_X5vUuBynKksF3xt6o_-IsoapF?usp=drive_link) who's shortcut must be in the location /Drive/MyDrive/data in your google drive folder.

1. VSLNet on Omnivor Features: 
    The code fetches the model from original Github repository. 
    Fetches the features directory from google drive. 

2. VSLNet on EgoVLP Features: 
    The code fetches the model and preprocessed prepared EgoVLP Features stored on the google bucket from previous run.
    Thus, no preparing of the dataset is required.  

3. VSLBase on Omnivor Features: 
    The code fetches the VSLNet model along with preprocessed prepared Omnivor Features from google bucket. 
    It is important to remove the directory VSLNet in the Episodic Memory directory, and replace it with VSLBase. 

4. VSLBase on EgoVLP Features:
    The code fetches the VSLNet model along with preprocessed prepared EgoVLP Features from google bucket. 
    It is important to remove the directory VSLNet in the Episodic Memory directory, and replace it with VSLBase. 

5. VSLNet on EgoVLP Features and GloVe Embeddings. 
    The code fetches the model from original Github repository. 
    Fetches the features directory and the GloVe embeddings from google drive.  
