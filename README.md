# Neural-Machine-Translation-with-Bahdanau-Attention

In this repo, we will deal with Translation of languages, specifically English to French. An Encoder-Decoder model using LSTM layers combined with Bahdanau Attention Mechanism will be implemented for training (along with Teacher Forcing - feeding the translation in a sequential manner to the decoder). Teacher Forcing can be compared to spoon-feeding wherein we ask the decoder model to spit out the next word when fed with the previously translated word(s) as well as the complete sequence of input language words (weighted with attention)

After the model gets trained, we will look at two inference algorithms viz. Greedy Search and Beam Search and discuss about each. For model's evaluation, we propose the BLEU metric to know how good the model can translate (albeit the metric having its own limitations)

Machine Translation models have been in existence for a long time. They are composed of Encoder-Decoder linked models. Encoder model's primary objective is to encode and learn the sequences of the source language while the Decoder Model's objective is to translate it to a target language. While these models have worked good, Bahdanau et. al. in circa 2014 came up with an Attention Mechanism that helps the decoder to focus on some words from the source language while translating and spitting out each word. This led to remarkable improvements in the model's performance and now Attention is being implemented to several applications apart from translation work in domains such as Computer Vision, etc.

In this notebook, we need to have tensorflow (2.4) installed in order to run it. Some knowledge on Deep NN Models is assumed as well as knowledge on RNN-LSTM networks

Source Language:- English
Target Language:- French

The dataset is provided by an organization through the below link:
Link for getting data - http://www.manythings.org/bilingual/

The notebook is segmented into sections:

Section 1: Data Processing <br>
Section 2: Data Tokenization <br>
Section 3: Defining the Model <br>
Section 4: Training the Model <br>
Section 5: Inference from the Model (Greedy & Beam Search) <br>
Section 6: Data Processing


## Section 1: Data Processing




## Section 2: Data Tokenization


## Section 3: Defining the Model


## Section 4: Training the Model


## Section 5: Inference from the Model (Greedy & Beam Search)



## Section 6: Data Processing
