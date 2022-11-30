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

## Method:
* Input:
  * For BATCH SIZE = 1
  * Original SCOP code takes input in (128,128) resolution. I have tried to keep the same. Didn't try higher resolution as it can increase the GPU overhead. Also, results shouldn't be much impacted by it.
  * Using the normalization using pre-specified value in the SCOPS code(which is of IMAGE NET).

* Encoder:
  * Input (128,128) is passed through 4 layers of RESNET with output: (1, 2048, 17, 17)

* Slot
  * Position encoding is applied on permuted result : [1, 17, 17, 2048] of size (17, 17)
  * This is then flattened to get the set for slot attention of vector size 2048. Shape is [1, 289, 2048].
  * Reduced the dimensionality of slot input using a linear layer to 64 to match the one specified in paper. Also working under the assumption smaller dimensionality might work better.
  *  Size of slots is [1, 6, 64]. Here 6 represents 5 parts and one for background. Javing ine extra slot always for background.

* Decoder
  * Decoder needs an input such as (batch, h, w, channels).
  * For this we treat each slot as a batch and create a 1*1 window of 64 channels by reshaping the [1, 6, 64] to [1*6, 1, 1, 64]. 
  * The 1*1 window is then repeated 8*8 times to create final decoder input : [6, 8, 8, 64].
  * Decoder positional encoding is used of size (8,8).
  * Decoder positional encoding output is permuted to [6, 64, 8, 8]
  * After which we have used the decoder in SCOPS and not in the SLOT attention code.
  * Decoder output would be [6, 1, 8, 8] 
  * We have used 1 channel as final output which will represent the part mask corresponding each slot.

* Final output:
  * Decoder output is then reshaped to [1, 6, 8, 8] which represnts batch size = 1, num of channels = 6 (one corresponding to background and others for each part), and window size of (8, 8)
  * Interpolation to get to the original size: [1, 6, 128, 128]

* VGG output:
  * The image is normalized according to VGG norms.
  * Passed through VGG19 architecture and input is extracted at relu5_4 layer. Output dim is [1, 8, 8, 512] which is interpolated to [1, 512, 128, 128].

* Semantic loss:
  * Softmax of prediction is taken along dimension 1: [1, 6, 128, 128]
  * Remove the first channel assuming it to be background. New pred : [1, 5, 128, 128]
  * Prediction is flatten : [1*128*128, 5]
  * VGG prediction is flattened: [1*128*128, 512]
  * Prediction output is changed to 512 dim using part basis generator which has a Parameter of shape [5, 512].
  * After which MSE loss is calculated between Prediction([1*128*128, 512]) and VGG features([1*128*128, 512]).

  


