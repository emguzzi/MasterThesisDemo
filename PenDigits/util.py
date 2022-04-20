import numpy as np
## taken from git repository of 'Anomaly detection in streamed data'
def load_pendigits_dataset(filename):
    with open(filename, 'r') as f:
        data_lines = f.readlines()

    data = []
    data_labels = []
    current_digit = None

    for line in data_lines:
        if line == "\n":
            continue

        if line[0] == ".":
            if "SEGMENT DIGIT" in line[1:]:
                if current_digit is not None:
                    data.append(np.array(current_digit))
                    data_labels.append(digit_label)

                current_digit = []
                digit_label = int(line.split('"')[1])
            else:
                continue

        else:
            x, y = map(float, line.split())
            current_digit.append([x, y])
            
    data.append(np.array(current_digit))
    data_labels.append(digit_label)

    return data, data_labels
