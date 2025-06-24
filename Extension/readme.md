The extension part is divided into ____ part. 

Requirements: Required dataset folder (universal for all notebooks) (https://drive.google.com/drive/folders/1a9Tfdi_X5vUuBynKksF3xt6o_-IsoapF?usp=drive_link) who's shortcut must be in the location /Drive/MyDrive/data in your google drive folder. 

1. Pretraining on synthetic dataset with 10% Validation HoldOut:
      The respective notebook (Pretraining-On-Whole-Synthetic_dataset) contains the necessary code block. The VSLNet model (available in the folder "VSLNetForPreTraining - With 10% Validation HoldOut") was modified to only accept training and validation set. It only saves top 1 model.

2. Pretraining on whole synthetic dataset:
       The respective notebook (Pretraining-On-Synthetic-Dataset-with-10%-Validation-HoldOut) contains the necessary code block. The VSLNet model (available in the folder "VSLNetForPreTraining - With whole synthetic dataset for training") was modified to only accept training set. It only saves top 1 model.

3. Transfer Learning VSLNet on original Ego4D Dataset:
       The respective notebook (Step2-TransferLearningOnOriginalEgo4D) contains the necessary code block. The VSLNet model (available in the folder "Pretrainable VSLNet") was modified to expect a .t7 file that would contain the pretrained weights.

4. Transfer Learning VSLNet on original Ego4D Dataset - With Frozne Feature Encoder:
       The respective notebook (Step2-TransferLearningOnOriginalEgo4D) contains the necessary code block. The VSLNet model (available in the folder "Pretrainable VSLNet - (But with Feature Encoder Frozen)") was modified to expect a .t7 file that would contain the pretrained weights.
