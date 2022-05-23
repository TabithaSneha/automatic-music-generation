# Automatic Music Generation

***

In this project, the Automatic Music Generating model has been created using Deep Learning, where LSTMs predict musical notes by analyzing a combination of given notes fed as input.

## Introduction:-

Long Short Term Memory(LSTM) Network is an advanced Recurrent Neural Network(RNN), a sequential network, that allows information to persist. LSTM was intoduced to solve certain drawbacks of RNN. RNNs were absolutely incapable of handling such “long-term dependencies”, ie, they failed to store information for a longer period of time. There was also no finer control over which part of the context needed to be carried forward and how much of the past needed to be ‘forgotten’. Other issues with RNNs are vanishing gradients which occur during the training process of a network through backpropagation, is taken care of by the LSTM network.

### Architecture of an LSTM Network:-

LSTMs deal with both Long Term Memory (LTM) and Short Term Memory (STM) and for making the calculations simple and effective it uses the concept of gates.
* **Forget Gate**: LTM goes to forget gate and it forgets information that is not useful.
* **Learn Gate**: Event(current input) and STM are combined together containing information that is recently learnt and forgetting any unnecessary information.
* **Remember Gate**: LTM information that hasn't been forgotten and the STM and Event are combined together in Remember gate which outputs a new updated LTM.
* **Use Gate**: This gate also uses information from LTM, STM, and Event to predict the output of a newly updated STM.

<img src="https://i.imgur.com/xdZnDWX.png" height="300" width="550" >

Image Source: Udacity

## Methodology:-

### Libraries used:-

* Tensorflow 2.2 - Open-source library for Machine Learning and AI.
* music21 - Python library used to parse and read various musical files. Here, Musical Instrument Digital Interface(MIDI) files have been used.
* Sklearn - Library for Machine Learning algorithms.
* NumPy
* Matplotlib

### Dataset used:-

The dataset used consists of Classical Piano MIDI files containing compositions of 19 famous composers from [Classical Piano MIDI Page](http://www.piano-midi.de/).

### Structure of Code:-

#### Reading the MIDI file:-

* Importing the required libraries given above.
* Calling each MIDI file from the directory and parsing the file.
* Separating all the instruments from the file and obtaining data only from Piano instrument.
* Iterating over all the parts of sub stream elements to check if element's type is Note or Chord. If it is Chord, split them into notes.
* Collecting all the notes in an array.

#### Input and Output Sequences for the Model:-
* Identifying the Unique Notes and finding the frequency of each such note.
* Filtering notes greater than a threshold frequency of 50 Hz.
* Creating a dictionary to convert note index to note and vice versa.
* Creating the input and output sequences for the model by using a timestep of 50.
* Reshaping input and output for the model and splitting the input into 80% for training and 20% for testing sets.

#### Creating the Model and Training the Neural Network:-
* Creating the model with two stacked LSTM layers with the latent dimension of 256 and a fully connected layer for the output with softmax activation.
* Compiling the model using Adam optimizer and training the neural network for 120 epochs.

#### Generating notes from the Trained Model:-
* Using the trained model, the notes will be predicted by generating a random index for the input array.
* Using the ‘np.argmax()’ function, we will get the data of the maximum probability value, which is converted to a note using the dictionary.
* Repeat the process to generate 200 notes.
* Saving the final predicted notes as a MIDI file.

### Architecture:-

Two stacked LSTM layers have been used with a dropout rate of 0.5. Dropout Layer is a regularization technique to reduce overfitting in deep learning models. A fully connected layer of size equal to the length of unique notes is used with 'softmax' activation(used for multi-class classification problems).

<img src="https://i.imgur.com/p2MScHw.png" height="350" width="750" >

| Parameters| Values| 
| -------- | -------- | 
| Optimizer     | Adam     |
| Batch Size     | 512     |
| Epochs     | 120     |

## Results:-

### Plot of Accuracy vs Epochs:-

<img src="https://i.imgur.com/Ovvp6yl.png" height="300" width="400" >

### Plot of Loss vs Epochs:-

<img src="https://i.imgur.com/VTsgjYf.png" height="300" width="400" >

| Dataset | Accuracy | Loss |
| -------- | -------- | -------- |
| Training     | 0.8257  | 0.5014     |
| Testing     | 0.5220     | 2.5357     |

You can listen to the predicted output music by downloading the MIDI files that have been uploaded above.

## References:-

[Udacity: Architecture of LSTM Network](https://classroom.udacity.com/courses/ud187/lessons/75c3cb92-67fb-4ef5-b0f3-5b56bd30bed9/concepts/a0f1d4bb-2c9a-4632-a89c-9ff96a1538cd)

[Data Flair Automatic Music Generation using Deep Learning](https://data-flair.training/blogs/automatic-music-generation-lstm-deep-learning/)
