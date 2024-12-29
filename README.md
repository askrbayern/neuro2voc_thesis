# neuro2voc_thesis
repository for the thesis neuro2voc

Author can be contacted at fei.gao@uzh.ch

Note: each experiment uses its own data, except CEBRA and VAE

Note: the codes are usually ordered, starting from 1. They should be executed in order. Only the last code in each folder is the model.

### 1_ML

1. Extract different features
2. Experiment 1: Basic ML
3. Experiment 2: Information integration

### 2_NLP

1. Extract data of interest
2. Use these data, create tensors
3. Create data from tensors
4. Train tokenizer based on text data
5. Train the model

### 3_DL

1. Extract tensor data
2. Concatenate the data from the tensor
3. Slice the tensors to prepare for the time series DL models
4. Batch processing

### 4_CEBRA

1. CEBRA-neuro2voc

### 5_VAE

1. clean the raw neural data, convert the spikes back to 30kHz or other Hz you want
2. use the above result, create training datasets
3. pass data into the models, train, and visualize

#### NeuralEnCodec
It is modified from https://github.com/marisbasha/neural_encodec

1. data
   1. dataAOI: data Area Of Interest, which means we only extract data from vocalization periods
   2. dataNotAOI: we take all data
2. model
   1. encAOI_binary: for binary data
   2. encAOI_binned: for binned data
3. visualize
   1. encAOI_visualize_binary
   2. encAOI_visualize_binned
