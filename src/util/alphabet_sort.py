# alphabet_sort.py - a program that sorts the contents of a text file alphabetically
# By: Ron Pascual

def sort_file_alphabetically(input_file, output_file):
    # Read all lines from the file
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Strip newline characters and sort
    sorted_lines = sorted(line.strip() for line in lines if line.strip())
    
    # Write sorted lines to the new file
    with open(output_file, "w", encoding="utf-8") as f:
        for line in sorted_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    sort_file_alphabetically("guesses.txt", "guesses_sorted.txt")
    print("Created guesses_sorted.txt with alphabetically sorted entries!")

