# SCOPS: SLOT Attention

Dataset - https://drive.google.com/file/d/1KmaqKL5-qjNIFM4UqwytRYkRQLLlZljN/view?usp=share_link

## Steps:
* Give dataset path in line 41 of train.py
* Run train.py
* Using deeplab.py by default. Model changes are added there
* Input file would be of size (224, 224)
* Only kept the semnatic loss for now
* Using different file for dataset creation(pascal_parts.py and lit_dataset.py). Ignore batch keys in dataloader other than batch['imgs']. Others are specific to SCOPS paper which we are not using here. 


## Issue:
* In deeplab.py, code works till line 241. Next step is to try and figure out what could be the input for decoder.
* For batch size = 2 and num_slots = 5, slots return output : (2, 5, 64)
* Need to modify it so that its a valid input for decoder in form of (batch, h, w, channel)
* In slot attention paper:
  * they have converted it to (2*5, 8, 8, 64) which is a seprate (8, 8, 64) map for all 10 slots.
* To be consistent to SCOPS paper, we have to figure out an ouput in form of (batch, num_parts+1, h, w). num_channels here is num_parts+1.
* Each channel is supposedly predicting a part, while in SLOT paper they were trying to reconstruct the original image with decoder.

## Note:
* Can try to predict each slot as a part map. 


## MODIFICATIONS
- I have create some changes in the new branch concentration_loss. I changed everything to get output of [batch_size * num_slots, Channel, h, w] from the last output.
- I had some problems with semantic consistency loss, so I tried to use concentration_loss.
- Every tensor and model is on CPU currently. If you want to run this code and see the output you can.
- I have no definite ideas as of yet for handling slots and combining them.
