import random


def load_dictionary(file_path):
    with open(file_path) as file:
        words = [line.strip() for line in file]
    return words

def is_valid_guess(guess, guesses):
    return guess in guesses

def evaluate_guess(guess, target_word):
    feedback = []
    str = ""
    for i in range(5):
        if guess[i] == target_word[i]:
            str += "\033[32m" + guess[i] + "\033[0m"  # Green
            feedback.append(('green', i, guess[i]))
        elif guess[i] in target_word:
            str += "\033[33m" + guess[i] + "\033[0m"  # Yellow
            feedback.append(('yellow', i, guess[i]))
        else:
            str += "\033[30m" + guess[i] + "\033[0m"  # Black
            feedback.append(('black', i, guess[i]))
    return str, feedback

def wordle(guesses, answers):
    target_word = random.choice(answers)
    attempts = 1
    max_attempts = 6
    remaining_guesses = answers.copy()
    greens = [None] * 5
    yellows = []
    blacks = set()

    def filter_guesses(guesses, greens, yellows, blacks):
        filtered = []
        for word in guesses:
            valid = True
            # Green letters must match
            for i, g in enumerate(greens):
                if g and word[i] != g:
                    valid = False
                    break
            # Yellow letters must be present but not in the same position
            for y in yellows:
                if y[2] not in word or word[y[1]] == y[2]:
                    valid = False
                    break
            # Black letters must not be present
            for b in blacks:
                if b in word:
                    valid = False
                    break
            if valid:
                filtered.append(word)
        return filtered

    while attempts <= max_attempts:
        filtered_guesses = filter_guesses(remaining_guesses, greens, yellows, blacks)
        if not filtered_guesses:
            print("No possible guesses left!")
            break
        guess = random.choice(filtered_guesses)
        remaining_guesses.remove(guess)
        #print(f"Attempt {attempts}: Bot guesses '{guess}'")

        if not is_valid_guess(guess, guesses):
            print(f"Invalid guess: {guess}")
            continue

        result, feedback = evaluate_guess(guess, target_word)
        #print(f"Result: {result}")

        # Update greens, yellows, blacks
        for color, idx, letter in feedback:
            if color == 'green':
                greens[idx] = letter
            elif color == 'yellow':
                # Avoid duplicate yellow entries
                if (color, idx, letter) not in yellows:
                    yellows.append((color, idx, letter))
            elif color == 'black':
                blacks.add(letter)

        x = {
            "greens": greens.copy(),
            "yellows": yellows.copy(),
            "blacks": list(blacks)
        }

        y = guess

        with open("training_data.txt", "a") as f:
            f.write(str(x) + " -> " + y + "\n")

        if guess == target_word:
            #print(f"Congratulations! The bot guessed the word: {target_word}")
            break

        attempts += 1
    else:
        print(f"Sorry, the bot did not guess the word. The word was: {target_word}")

guesses = load_dictionary('guesses.txt')
answers = load_dictionary('answers.txt')

for i in range(1000):
    wordle(guesses, answers)