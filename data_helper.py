import os
import re
import numpy as np


def write_csv_file(text_dirs, labels, output_dir, file_name, max_data_num=None):
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    s = ''
    for label, path in zip(labels, text_dirs):
        with open(path, 'r', encoding='utf8', errors='ignore') as f:
            all_lines = f.readlines()
        if max_data_num is not None and len(all_lines) > max_data_num:
            all_lines = all_lines[:max_data_num]
        fix = np.array([str(label) + ',,'] * len(all_lines))
        all_lines = np.array(all_lines)
        s = s + ''.join(np.core.defchararray.add(fix, all_lines).tolist())
        # for line in all_lines:
        #     s = s + str(label) + ',,' + line
    with open(os.path.join(output_dir, file_name), 'w', encoding='utf8', errors='ignore') as f:
        f.write(re.sub(r'\bunknown\b', '<unk>', s))


def write_csv_files(train_text_dirs, test_text_dirs, train_labels, test_labels, output_csv_dir, train_file, test_file,
                    max_data_num=None, max_test_data_num=None):
    write_csv_file(train_text_dirs, train_labels, output_csv_dir, train_file, max_data_num)
    write_csv_file(test_text_dirs, test_labels, output_csv_dir, test_file, max_test_data_num)
