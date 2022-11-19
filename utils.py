def tag_to_idx(filepath):
    idx_dict = {'': 0}
    idx_dict_reverse = {'0': ''}
    id_to_assign = 1
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
    idx_dict = {'': 0}
    for filepath in filepath_list:
        id_to_assign = 1
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

def dataset_build_with_batch(filepath, batch_size):
    training_data = []
    f = open(filepath, "r")
    lines = f.readlines()
    f.close()
    sentence = []
    tags = []
    max_sentence_length = 0
    for line in lines:
        if len(line) > 4:
            word = line.split(' ')[0]
            tag = line.split(' ')[3][:-1]
            sentence.append(word)
            tags.append(tag)
        else:
            if len(sentence) > max_sentence_length:
                max_sentence_length = len(sentence)
            training_data.append((sentence, tags))
            sentence = []
            tags = []


    for item in training_data:
        old_len = len(item[0])
        for i in range(old_len, max_sentence_length):
            item[0].append('')
            item[1].append('')

    training_data_new = []
    sentence = []
    tags = []
    for i, item in enumerate(training_data):
        for word in item[0]:
            sentence.append(word)
        for tag in item[1]:
            tags.append(tag)
        if (i + 1) % batch_size == 0:
            training_data_new.append((sentence, tags))
            sentence = []
            tags = []

    if len(sentence) > 0:
        training_data_new.append((sentence, tags))

    return training_data_new

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
    print(dataset_build_with_batch("conll2003/train.txt", 32))