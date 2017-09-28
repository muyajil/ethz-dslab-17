# Module Requirements

## Definitions:

- Debug: Validate that the code is running. Run using 1 training and 1 validation batch
- Validate: Validate some hyperparameters. Run using the full dataset, according to some split ratio.
- Train: Train a model on the full dataset and save the model.
- Compress: Use a trained model to compress pictures
- Decompress: Decompress images using model 


## Main.py:

- Handles the execution 
- Stuff to handle:
-- Debug vs. Validate vs. Production run
-- Which data source
-- Split Ratio (train/test, full train means test is empty)
-- Parsing of Command Line Arguments
-- Model saving/restoring
-- Evaluation of metrics 
-- Tensorboard

##Encode.py

- Takes a model and outputs compressed images

## Decode.py

- Takes a model and outputs decompressed images

## Dataset:

- Class that represents the dataset
- Methods to export:
-- Get_next_batch
-- Split (Input: split ratio. Output: 2 Datasets)
-- Constructor (Input: Data Location, Naming Scheme, Transformation/Augmentation Pipeline. Output: Dataset)
-- When debug flag is given, we only use 2 batches
-- The dataset class should support building an augmentation and transformation pipeline. Something in the form of passing a list of functions that augment and transform the data. Here the question is how do we validate that the output of the nth function fits into the input of the n+1st function?
-- Augmentation Function (Input: 1 Datapoint, Output: Multiple Datapoints)
-- Transformation Function (Input: 1 Datapoint, Ouput: 1 Datapoint)

## Model:

- Abstract class
- Each model needs to derive from this.
- Methods to export:
-- Compress (Input: Datapoint. Output: Compression)
-- Decompress (Input: Compressed Datapoint. Output: Reconstructed Datapoint)
-- Train (Input: Training Data. Output: Metrics)
-- Constructor (Input: Running Mode, Config, *Model Path. Output: Model)
