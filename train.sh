python3 -u train.py\
    --data_name "cavity60k" \
    --input_path "datasets/sample/cavity60k/10000_train_f1.h5" \
    --output_path "train" \
    --maxmin_path "datasets/raw/cavity60k.csv" \
    --batch_size 1 \
    --max_epoch 200 \
    --display_verbose 1 \
    --progress_freq 100 \
    --display_freq 4000 \
    --summary_freq 100 \
    --save_freq 1000
    
