## Probabilistic Shape Completion with Multi-target Conditional Variational Autoencoders


This repository accompanies the article expected to be published in 
the 22nd International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2019), 
which will be held from October 13th to 17th, 2019, in Shenzhen, China.

Please consider citing the draft of the [paper on arxiv](https://arxiv.org/pdf/1906.11957.pdf) if you enjoyed the implementation:

    @misc{1906.11957,
    Author = {Amir H. Abdi and Mehran Pesteie and Eitan Prisman and Purang Abolmaesumi and Sidney Fels},
    Title = {Variational Shape Completion for Virtual Planning of Jaw Reconstructive Surgery},
    Year = {2019},
    Eprint = {arXiv:1906.11957},
    }

### Download data
To download the data and set the environment variable $DATASETS to where the data is 
downloaded, run

    source download-data.sh 


### Train and Test Model
This is a Python3 implementation. To train the conditional VAE model for shape completion with the default data 
(mandible dataset), install the requirements by running

    pip install -r requirements.txt

And run the training script
    
    bash scripts/train-CVAE-vwDice-TWcvae
    

To test the model, set the `--test=true` and set the 
`--load_model_path` flag to where the trained model is stored. 


### Sample Results

![Reconstructed Samples](./imgs/TestCases.png)