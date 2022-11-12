def tag_to_idx(filepath):
    idx_dict = {}
    idx_dict_reverse = {}
    id_to_assign = 0
    f = open(filepath, "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        if len(line) > 4:
            label = line.split(' ')[3][:-1]
            if label not in idx_dict:
                idx_dict[label] = id_to_assign
                idx_dict_reverse[str(id_to_assign)] = label
                id_to_assign += 1

    return idx_dict, idx_dict_reverse

def word_to_idx(filepath_list):
    idx_dict = {}
    for filepath in filepath_list:
        id_to_assign = 0
        f = open(filepath, "r")
        lines = f.readlines()
        f.close()
        for line in lines:
            if len(line) > 4:
                label = line.split(' ')[0]
                if label not in idx_dict:
                    idx_dict[label] = id_to_assign
                    id_to_assign += 1

    return idx_dict

def dataset_build(filepath):
    training_data = []
    f = open(filepath, "r")
    lines = f.readlines()
    f.close()
    sentence = []
    tags = []
    for line in lines:
        if len(line) > 4:
            word = line.split(' ')[0]
            tag = line.split(' ')[3][:-1]
            sentence.append(word)
            tags.append(tag)

        else:
            training_data.append((sentence, tags))
            sentence = []
            tags = []

    return training_data

def summary(filepath):
    idx_dict = {}
    idx_dict['total'] = 0
    id_to_assign = 0
    f = open(filepath, "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        if len(line) > 4:
            label = line.split(' ')[3][:-1]
            if label not in idx_dict:
                idx_dict[label] = 1
            else:
                idx_dict[label] += 1

            idx_dict['total'] += 1

    return idx_dict


if __name__ == "__main__":
    print(summary("/Users/kasperxiaomingshen/Downloads/conll2003/train.txt"))