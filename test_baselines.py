import argparse
import torch
import pandas as pd
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


# my files
from common_utils import set_seed, process_results_file, valid_results_file_to_metrics
from property_handler import rdkit_no_error_print, property_init

def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='IPCA'
    )
    # end-end model settings
    parser.add_argument('--check_testset', default=True, action='store_true', help='get test results')
    parser.add_argument('--property', type=str, default='M', help='name of property to translate (should be folder with that name inside dataset)')
    parser.add_argument('--baseline', type=str, default='IPCA', help='testset filename')
    parser.add_argument('--SR_similarity', type=int, default=0.3, help='minimal similarity for success')
    parser.add_argument('--SR_property_QED_val', type=int, default=0.7, help='minimal property value for success')
    parser.add_argument('--SR_property_DRD2_val', type=int, default=0.3, help='minimal property value for success')
    parser.add_argument('--seed', type=int, default=50, help='base seed')

    args = parser.parse_args()
    return args


# check testset
def check_testset(args):
    results_file_path = 'baselines_outputs/' + args.property + '/' + args.baseline + '_' + args.property + '.txt'
    valid_results_file_path = 'baselines_outputs/' + args.property + '/valid_' + args.baseline + '_' + args.property + '.txt'
    print(' ')
    print('Loading results for model => ' + args.baseline)
    print('From file => ' + results_file_path)

    # train set for novelty
    if args.baseline is 'HG2G':
        train_set_file = 'baselines_outputs/' + args.property + '/' + args.baseline + '_train_pairs.txt'
        trainset_df = pd.read_csv(train_set_file, header=None, delimiter=' ')[[0,1]]
        trainset = set(trainset_df[0]).union(set(trainset_df[1]))
    elif args.baseline in ('JTVAE', 'IPCA'):
        train_set_file = 'baselines_outputs/' + args.property + '/' + args.baseline + '_train.txt'
        trainset_df = pd.read_csv(train_set_file, header=None)
        trainset = set((trainset_df.squeeze()).astype(str))
    else:
        print('unsupported model')

    # result file -> valid results + property and similarity for output molecules
    process_results_file(results_file_path, args, valid_results_file_path, trainset)
    testset_len = 780

    # calculate metrics
    validity_mean, validity_std, \
    diversity_mean, diversity_std, \
    novelty_mean, novelty_std, \
    property_DRD_mean, property_DRD_std, \
    property_QED_mean, property_QED_std, \
    similarity_mean, similarity_std, \
    SR_mean, SR_std = \
                        valid_results_file_to_metrics(valid_results_file_path, args, testset_len)


    # print results
    print(' ')
    print('Property => ' + args.property)
    print('Baseline => ' + args.baseline)
    print('DRD2 => mean: ' + str(round(property_DRD_mean, 3)) + '   std: ' + str(round(property_DRD_std, 3)))
    print('QED => mean: ' + str(round(property_QED_mean, 3)) + '   std: ' + str(round(property_QED_std, 3)))
    print('fingerprint Similarity => mean: ' + str(round(similarity_mean, 3)) + '   std: ' + str(round(similarity_std, 3)))
    print('SR => mean: ' + str(round(SR_mean, 3)) + '   std: ' + str(round(SR_std, 3)))
    print('validity => mean: ' + str(round(validity_mean, 3)) + '   std: ' + str(round(validity_std, 3)))
    print('novelty => mean: ' + str(round(novelty_mean, 3)) + '   std: ' + str(round(novelty_std, 3)))
    print('diversity => mean: ' + str(round(diversity_mean, 3)) + '   std: ' + str(round(diversity_std, 3)))


if __name__ == "__main__":

    with torch.no_grad():
        # parse arguments
        args = parse_arguments()

        # set seed
        set_seed(args.seed)

        # set device (CPU/GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        # initialize property value predictor
        property_init('DRD2')

        # disable rdkit error messages
        rdkit_no_error_print()

        # check testset
        if args.check_testset is True:
            check_testset(args)