# Wordle-AI-Solver

## About

An AI that plays New York Times' Wordle game at a human-level performance (~4 guesses on average). It was trained on 100,000+ simulated Wordle games and consistently solves ~93% of games within 6 guessses.

## Features
- wordle_game.py - allows user to play Wordle manually
- wordle_with_ai.py - watch the ai solve wordle
- train_ai_wordle.py - train the ai
- simulate_wordle.py - generate wordle games to be used for training datasets
- training_converter.py - convert datasets into proper training format (.json)

## Results
- Accuracy: ~93% solved within 6 attempts
- Average guesses per game: ~4.0 (comparable to human performance)

## How to Install
1. Clone the repo:
git clone https://github.com/RonPascual/Wordle-AI-Solver.git
cd Wordle-AI-Solver

## Requirements
Python 3.8+
PyTorch
NumPy

## How to Use
Training the AI
1. Run simulate_wordle.py to simulate Wordle games to be used for the training dataset
2. Run training_converter.py to convert the training dataset from training_data.txt to dataset.json to be used for AI training
3. Run train_ai_wordle.py to train the AI model based on the dataset adjusting epochs and batches as one sees fit

Testing the AI
1. Run wordle_with_ai.py to watch how well the AI does at solving wordle games
## Important Notes
- the full dataset is not included in this repo
