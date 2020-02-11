# FluidGAN-TensorFlow

![img1](https://github.com/jcl2018/cGAN-transport-phenomena/blob/master/img/cGAN-2019-015.jpg) 

FluidGAN-TensorFlow is a Tensorflow implementation of FluidGAN which is  a generic framework for predicting multi-physcis phenomena like convective transport and for both stationary and time-dependent settings. This framework is based on the conditional generative adversarial network (cGAN) [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow). 

Sample dataset is available once the paper is published.
More information about CMU Mechanical and AI LAB (MAIL) can be found [here](https://sites.google.com/view/barati).

## Getting started

Starting training in a few steps.
### Prerequisites

+ Tensorflow1 version >= tf.1.10
+ tftables
+ tqdm

### Recommended 
+ Linux with Tensorflow GPU edition + cuDNN

### Download Example Dataset
You can the raw dataset (in hdf5 format) inside sample_dataset/raw

	- sample_dataset
	  - raw
	  - shuffle
	  - sample

Before training, you should run the pre-processing script to get shuffled and sampled dataset(using k-fold cross validation).

### Train, Test 
To train the model using pre-defined parameters, simply run:

	bash ./train.sh

Similarly, run the test script with pre-defined parameters:

	bash ./test.sh

## Configuration details
In this section we provide some illustrations of the configuration parameters we defined. 

### Pre-processing
In data pre-processing step you will compute the extremums, shuffle the dataset and extract data samples for training, testing and predicting. You could first configure the "pre_process.sh" file and then run the shell file in your terminal:

    sh pre_process.sh

Here's the instruction of how to configure your shell file, and some examples for implementing each mode. After pre-processing you should able to find your sample file in "./dataset/sample/<dataset_name>/<mode>".

#### Configurations 
- dataset_name: The name of your dataset in "./dataset/raw" path.
- sample_num: The number of your data samples. In "train" mode, it's the total number of your training and testing set. In "test" mode, it's the total number of your testing set. This option is invalid in "movie" mode.
- fold_num: The number of fold for cross validation, default value is 5.
- mode: Can be chosen from "train", "test" and "movie". The train mode is selected when you want to train a model and test the performance. The test mode is used for only testing the model performance. In the "train" and "test" mode you can randomly extract the input and target data. The movie mode is enabled only for time-dependent dataset. It will extract data with same boundary / initial condition, but different time stamp. The input and target data will later be used to make the movie.
- shuffle_raw: Whether or not to shuffle your dataset before sampling. 0 to disable this option and 1 to enable it. Invalid for "movie" mode.
- chunk: Whether or not to apply "chunk". When processing large and high-dimensional dataset, sometimes you will encounter memory error in the shuffling and sampling process. By specifying chunk limit you can limit the maximum number of samples to avoid the memory error. It is used in the shuffling and sampling process. If your code raised memory error in the shuffling process, chunk is automatically enabled. 0 to disable this option and 1 to enable it. Invalid for "movie" mode.
- chunk_limit: Maximum number for a chunk, valid when chunk is enabled.
- movie_num: Number of movie samples used for creating time-dependent flow movie, valid only for "movie" mode.
#### Examples
- Train mode
  - Dataset without chunk
  
        dataset_name="cavity60k"
        sample_num=200
        fold_num=5

        python -u pre_process.py \
            --dataset_name ${dataset_name} \
            --sample_num ${sample_num} \
            --fold_num ${fold_num} \
            --shuffle_raw 1 \
            --mode "train" \
            --chunk 0         

  - Dataset with chunk
  
        dataset_name="cavity_time3M"
        sample_num=100000
        fold_num=5
        
        python -u pre_process.py \
            --dataset_name ${dataset_name} \
            --sample_num ${sample_num} \
            --fold_num ${fold_num} \
            --shuffle_raw 1 \
            --mode "train" \
            --chunk 1 \
            --chunk_limit 50000
            
- Test mode
  - Dataset without chunk 
  
        dataset_name="cavity60k"
        sample_num=200
        fold_num=5

        python -u pre_process.py \
            --dataset_name ${dataset_name} \
            --sample_num ${sample_num} \
            --fold_num ${fold_num} \
            --shuffle_raw 1 \
            --mode "test" \
            --chunk 0      
- Movie mode
  
        dataset_name="cavity_time3M"

        python -u pre_process.py \
            --dataset_name ${dataset_name} \
            --mode "movie" \
            --movie_num 5 
 
### Training
The "train.py" contains the whole training process and "train.sh" provides an example to start the training.
#### Configurations 
- input_file: input h5 file for training.
- output_dir: output directory consisting of model data and real time output
- csv_path: a csv file containing the extremums, used to scale the output
- mode: must be "train" in this step.
- summary_freq: update summary to tensorboard every summary steps.
- progress_freq: print running losses to terminal every progress steps.
- display_freq: save current input/output/target image to the local computer every display steps.
- save_freq: 

#### Example 

        dataset_name="cavity60k"
        sample_num=200
        fold_num=5

        for fold in $( seq 1 ${fold_num} )
        do
            python -u cfd2cfd_working.py \
                --input_file "dataset/sample/${dataset_name}_train/${sample_num}_f${fold}_train.h5" \
                --output_dir "train/${dataset_name}/${sample_num}_f${fold}_train" \
                --csv_path "dataset/raw/${dataset_name}.csv" \
                --mode "train" \
                --summary_freq 100 \
                --progress_freq 50 \
                --display_freq 5000 \
                --save_freq 5000 \
                --max_epochs 50 \
                --batch_size 1 \
                --channel 4 
        done 



