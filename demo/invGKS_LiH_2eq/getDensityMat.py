import sys
def process_matrix(input_filepath, output_filepath):
    """
    Reads a matrix from a file, multiplies each element by 0.5,
    and writes the new matrix to another file.

    Args:
        input_filepath (str): The path to the input file containing the matrix.
        output_filepath (str): The path to the output file where the processed matrix will be written.
    """
    processed_matrix = []

    try:
        with open(input_filepath, 'r') as infile:
            for line in infile:
                # Split the line into elements, convert to float, and multiply by 0.5
                # Handles both space-separated and comma-separated values
                elements = [float(x) * 0.5 for x in line.strip().replace(',', ' ').split()]
                processed_matrix.append(elements)

        with open(output_filepath, 'w') as outfile:
            for row in processed_matrix:
                # Write each processed row to the output file, elements separated by spaces
                outfile.write(' '.join(map(str, row)) + '\n')

        print(f"Matrix processed and saved to {output_filepath}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.")
    except ValueError:
        print("Error: Invalid data in the input file. Ensure all elements are numbers.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

inputFile = str(sys.argv[1])
outputFile = str(sys.argv[2])
process_matrix(inputFile, outputFile)
