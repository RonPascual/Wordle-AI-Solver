import random


def load_dictionary(file_path):
    with open(file_path) as file:
        words = [line.strip() for line in file]
    return words

def is_valid_guess(guess, guesses):
    return guess in guesses

def evaluate_guess(guess, target_word):
    str = ""

    for i in range(5):
        if guess[i] == target_word[i]:
            str += "\033[32m" + guess[i] + "\033[0m"  # Green
        elif guess[i] in target_word:
            str += "\033[33m" + guess[i] + "\033[0m"  # Yellow
        else:
            str += "\033[30m" + guess[i] + "\033[0m"  # Black
    
    return str

def wordle(guesses, answers):
    target_word = random.choice(answers)
    attempts = 1
    max_attempts = 6
    remaining_guesses = guesses.copy()

    while attempts <= max_attempts:
        guess = random.choice(remaining_guesses)
        remaining_guesses.remove(guess)
        print(f"Attempt {attempts}: Bot guesses '{guess}'")
        
        if not is_valid_guess(guess, guesses):
            print(f"Invalid guess: {guess}")
            continue

        result = evaluate_guess(guess, target_word)
        print(f"Result: {result}")

        if guess == target_word:
            print(f"Congratulations! The bot guessed the word: {target_word}")
            break

        attempts += 1
    else:
        print(f"Sorry, the bot did not guess the word. The word was: {target_word}")

guesses = load_dictionary('guesses.txt')
answers = load_dictionary('answers.txt')

wordle(guesses, answers)