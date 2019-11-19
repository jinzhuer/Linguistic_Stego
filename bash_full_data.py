import argparse
import os
import multiprocessing as mp
from build_dict import build_dict

log_dir = './log_attmask_icics_l0.0001'
os.makedirs(log_dir, exist_ok=True)


def make_task(GPU, data_folder, data_type, labeled_data_num, test_data_num, labels, bi_directional):
    bi_directional_str = 'bi_directional' if bi_directional.lower() in (
        "yes", "true", "t", "1") else 'uni_directional'
    log_file = os.path.join(log_dir, data_folder + '_' + data_type + '_' + str(
        labeled_data_num) + '_' + bi_directional_str + '_'.join([str(x) for x in labels]) + '.txt')
    return 'python train.py --GPU=%s --data_folder=%s --data_type=%s  --labeled_data_num=%f --test_data_num=%f --labels %s --bi_directional=%s >%s 2>&1' % (
        GPU, data_folder, data_type, labeled_data_num, test_data_num, ' '.join([str(x) for x in labels]),
        bi_directional, log_file)


def do_task(task):
    print('begin: %s' % task)
    print(os.system(task))
    print('done: %s' % task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gpu", nargs='+', type=int, default=[4,5,6,7], help="gpu id to use")
    parser.add_argument("--gpu", nargs='+', type=int, default=[0,1,2,3], help="gpu id to use")
    parser.add_argument("--bi_directional", nargs='+', type=str, default=["False", 'True'],
                        help="whether to use bi-directional LSTM")
    parser.add_argument("--thread_num", type=int, default=4, help="process on each GPU")
    args = parser.parse_args()
    gpus = [str(x) for x in args.gpu]
    print(gpus)
    gpu_num = len(gpus)
    build_dict()
    tasks = [[] for i in range(gpu_num)]
    pools = [mp.Pool(processes=args.thread_num) for i in range(gpu_num)]
    i = 0
    # without pre-train
    for bi_directional in args.bi_directional:
        for labels in [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]:  # , [0, 1, 2, 3, 4, 5]]:
            for data_type in ['movie', 'news', 'tweet']:
                unlabeled_data_num = 0
                labeled_data_num = 0.8
                test_data_num = 0.2
                tasks[i].append(
                    make_task(gpus[i], 'ACL', data_type, labeled_data_num, test_data_num, labels, bi_directional))
                i = (i + 1) % gpu_num
            # for data_type in ['5w_4s', '5w_8s', '7w_4s', '7w_8s']:
            #     labeled_data_num = 3500
            #     test_data_num = 500
            #     tasks[i].append(
            #         make_task(gpus[i], 'shi', data_type, labeled_data_num, test_data_num, labels, bi_directional))
            #     i = (i + 1) % gpu_num

    for i in range(gpu_num):
        for task in tasks[i]:
            pools[i].apply_async(do_task, (task,))
    for i in range(gpu_num):
        pools[i].close()
        pools[i].join()
