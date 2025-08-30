# wordle_game.py - A program that allows the user to play Wordle
# By: Ron Pascual

import random

def load_dictionary(file_path):
    with open(file_path) as file:
        words = [line.strip() for line in file]
    return words

def is_valid_guess(guess, guesses):
    return guess in guesses

# check if letters are in correct spot
def evaluate_guess(guess, target_word):
    str = ""

    for i in range(5):
        if guess[i] == target_word[i]:
            str += "\033[32m" + guess[i] # Green
        elif guess[i] in target_word:
            str += "\033[33m" + guess[i] # Yellow
        else:
            str += "\033[30m" + guess[i] # Black

    return str + "\033[0m"

def wordle(guesses, answers):
    print("Welcome to Wordle! You have 6 chances to guess the 5-letter word!")
    target_word = random.choice(answers)
    
    attempts = 1
    max_attempts = 6

    while attempts <= max_attempts:
        guess = input("Enter Guess #" + str(attempts) + ": ").strip().lower()
        
        if not is_valid_guess(guess, guesses):
            print(f"Invalid guess: {guess}. Please try again.")
            continue

        if guess == target_word:
            print(f"Congratulations! You guessed the word: {target_word}")
            break

        feedback = evaluate_guess(guess, target_word)
        print(f"Result: {feedback}")
        attempts += 1

    else:
        print(f"Sorry, you did not guess the word. The word was: {target_word}")

guesses = load_dictionary("data\\guesses.txt")
answers = load_dictionary("data\\answers.txt")

wordle(guesses, answers)