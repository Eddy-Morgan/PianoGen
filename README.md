# 7343 Homework 1

In this homework, your goal is to implement and train two LSTM models, one called piano music composer and the other called critic.  

## Data
The piano data (in midi format) can be downloaded from: 
https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0-midi.zip
By default, when unzipped, the data will be put into a directory named "maestro-v1.0.0".

The file "midi2seq.py" contains a set of functions that help to process the midi data and convert the data to sequences of events.   
The file "model_base.py" contains the base classes that you should inherit when implementing the following model classes.

## Task 1: Critic
(Class "Critic" should be a subclass of the class CriticBase. You must use the exact class name.)
You should implement a multi-layer (2 or 3 layers) LSTM model in this class. The Model (the score function) takes a sequence of envents as input and outputs a score judging whether the piano music corresponding to the sequence is good music or bad music. A function to generate random music is provided in the "midi2seq.py". Use the function to create a collection of random piano plays as examples of bad music. Use the piano plays in the downloaded data as example of good music. (You don't need to use all the downloaded data. A sufficiently large subset will be enough.) Train the model in this class using both the good and the bad examples.    

## Task 2: Composer
(Class "Composer" should be a subclass of the class ComposerBase. You must use the exact class name.)
You should implement a multi-layer (2 or 3 layers) LSTM model in this class. When the compose member function is called, it should return a sequence of events. Randomness is require in the implementation of the compose function such that each call to the function should generate a different sequence. The function "seq2piano" in "midi2seq.py" can be used to convert the sequence into a midi object, which can be written to a midi file and played on a computer. Train the model as a language model (autoregression) using the downloaded piano plays.

## Task 3: Composer VS Critic 
Use your Critic model to score the music generated by your composer. Compose 50 music sequences and score them using the Critic. 
  - What is the average score of the generated music? 
  - Propose an approach to improve the quality of your composed music.
  - What major difficulty do you expect to encounter when using the above proposed approach?

## Submit your work
Develop and train your models for tasks 1 and 2. Also conduct experiment and provide answers to task 3. Put all your code and answers to questions in task 3 in a single file named "hw1.py" (*you must use this file name*) and submit the file in moodle. The answers in task 3 should be placed at the beginning of the file in a comment setion (use ''' to mark a multi-line comment section).   

We may test your implementation using code similar to the following (code for composer is show and we will test critic in a similar way):
    
    from hw1 import Composer
    piano_seq = torch.from_numpy(process_midi_seq())
    loader = DataLoader(TensorDataset(piano_seq), shuffle=True, batch_size=bsz, num_workers=4)
    
    cps = Composer()
    for i in range(epoch):
        for x in loader:
            cps.train(x[0].cuda(0).long())
            
    midi = cps.compose(100)
    midi = seq2piano(midi)
    midi.write('piano1.midi')

Note that the above code trains your model from scratch. In addition, you should provide trained weights for both your models. We may create your models by calling the constructor with "load_trained=True". In this case, your class constructor should: 
 - Download the trained weights from your google drive. (Do not upload the weights to moodle. Instead, you should store them on google drive.)
 - Load the trained weights into the model class object.

For example, if we do: m = Composer(load_trained=True), m should be a Composer model with the trained weights loaded. We should be able to call m.compose without training it and obtain a piano sequence from the downloaded trained model. 
 
 
