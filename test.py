import argparse
import torch
import numpy as np
from scipy.spatial import distance
import pandas as pd
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


# my files
from embedding_translator import load_checkpoint, load_checkpoint_unified
from embedder_train import get_dataloader
from data_preprocess import filname2testset, filname2testset_unified
from common_utils import set_seed, input2output, get_random_list, generate_results_file, process_results_file, \
    valid_results_file_to_metrics
from property_handler import rdkit_no_error_print, property_init, property_calc, similarity_calc, is_valid_molecule

def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='IPCA'
    )
    # end-end model settings
    parser.add_argument('--check_testset', default=True, action='store_true', help='get test results')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for end-end model')
    parser.add_argument('--property', type=str, default='M', help='name of property to translate (should be folder with that name inside dataset)')
    parser.add_argument('--test_direction', type=str, default='AB', help='direction of translation- AB: A->B; BA: B->A')
    parser.add_argument('--testset_filename', type=str, default='A_test.txt', help='testset filename')
    parser.add_argument('--checkpoints_folder', type=str, default='checkpoints', help='name of folder for checkpoints loading')
    parser.add_argument('--num_retries', type=int, default=100, help='number of retries for each test sample (K)')
    parser.add_argument('--SR_similarity', type=int, default=0.3, help='minimal similarity for success')
    parser.add_argument('--SR_property_QED_val', type=int, default=0.7, help='minimal property value for success')
    parser.add_argument('--SR_property_DRD2_val', type=int, default=0.3, help='minimal property value for success')
    parser.add_argument('--plots_folder', type=str, default='plots_output', help='name of folder for plots saving')
    parser.add_argument('--seed', type=int, default=50, help='base seed')
    parser.add_argument('--test_max_len', type=int, default=90, help='length of test smiles')
    parser.add_argument('--save_in_out_molecules', default=True, action='store_true', help='save a file with input-output molecules')

    # discover approved drugs
    parser.add_argument('--discover_approved_drugs', default=False, action='store_true', help='check model for discovering approved drugs')
    parser.add_argument('--FDA_approved_filename', type=str, default='dataset/FDA_approved_drugs_drugbank/FDA_approved_canon_clean.csv', help='filename of FDA approved drugs')
    parser.add_argument('--approved_drugs_retries', type=int, default=100, help='number of retries for each drug sample')
    parser.add_argument('--approved_drugs_max_len', type=int, default=90, help='length of approved drug smiles')

    # intermediate analyse
    parser.add_argument('--intermediate_analysis', default=False, action='store_true', help='analyse model intermediate outputs')
    parser.add_argument('--num_samples', type=int, default=10, help='number of samples')
    parser.add_argument('--num_neighbours', type=int, default=40, help='num neighbours for intermediate analyse')

    # ablation (old)
    parser.add_argument('--use_fp', default=True, action='store_true', help='does translator use molecule fp')

    # Ablation (new)
    parser.add_argument('--unified_destination', default=False, action='store_true', help='uses 1 B domain for all molecules')
    parser.add_argument('--no_embedding', default=False, action='store_true', help='zero the embedding during infernece, rely on fp')


    args = parser.parse_args()
    return args


# check testset
def check_testset(args, model_in, model_out_QED, model_out_DRD2, input2output_func):
    testset_path = 'dataset/' + args.property + '/' + args.testset_filename
    results_file_path = args.plots_folder + '/' + args.property + '/IPCA_' + args.property + '_test.txt'
    valid_results_file_path = args.plots_folder + '/' + args.property + '/valid_IPCA_' + args.property + '_test.txt'
    print(' ')
    print('Loading testset from file   => ' + testset_path)

    # build testset from filename
    if model_out_DRD2 is not None:  # regular
        testset_list = filname2testset(testset_path, model_in, model_out_QED, model_out_DRD2)
        trainset = set(model_in.dataset.trainset).union(set(model_out_QED.dataset.trainset), set(model_out_DRD2.dataset.trainset))
    else:
        testset_list = filname2testset_unified(testset_path, model_in, model_out_QED)
        trainset = set(model_in.dataset.trainset).union(set(model_out_QED.dataset.trainset))
    test_loader = get_dataloader(model_in, args, testset_list, batch_size=args.batch_size, shuffle=False)

    # generate random seeds
    random_seed_list = get_random_list(args.num_retries)

    def input2output_func_aux(input_batch):
        return input2output_func(input_batch, random_seed_list)

    def input2smiles(input):
        return model_in.tensor2string(input)  # to smiles

    # generate results file
    generate_results_file(test_loader, input2output_func_aux, input2smiles, results_file_path)

    # result file -> valid results + property and similarity for output molecules
    process_results_file(results_file_path, args, valid_results_file_path, trainset)

    # calculate metrics
    validity_mean, validity_std, \
    diversity_mean, diversity_std, \
    novelty_mean, novelty_std, \
    property_DRD_mean, property_DRD_std, \
    property_QED_mean, property_QED_std, \
    similarity_mean, similarity_std, \
    SR_mean, SR_std = \
                        valid_results_file_to_metrics(valid_results_file_path, args, len(testset_list))

    # print results
    print(' ')
    print('Property => ' + args.property)
    print('DRD2 => mean: ' + str(round(property_DRD_mean, 3)) + '   std: ' + str(round(property_DRD_std, 3)))
    print('QED => mean: ' + str(round(property_QED_mean, 3)) + '   std: ' + str(round(property_QED_std, 3)))
    print('fingerprint Similarity => mean: ' + str(round(similarity_mean, 3)) + '   std: ' + str(round(similarity_std, 3)))
    print('SR => mean: ' + str(round(SR_mean, 3)) + '   std: ' + str(round(SR_std, 3)))
    print('validity => mean: ' + str(round(validity_mean, 3)) + '   std: ' + str(round(validity_std, 3)))
    print('novelty => mean: ' + str(round(novelty_mean, 3)) + '   std: ' + str(round(novelty_std, 3)))
    print('diversity => mean: ' + str(round(diversity_mean, 3)) + '   std: ' + str(round(diversity_std, 3)))



def discover_approved_drugs(args, model_in, model_out, input2output_func):
    print(' ')
    print('Loading approved drugs dataset from file   =>' + args.FDA_approved_filename)
    # build testset from filename
    approved_drugs_list, smiles_name_dict = filname2testset(args.FDA_approved_filename, model_in, model_out, drugs=True)
    approved_drugs_list = [drug for drug in approved_drugs_list if len(drug) <= args.approved_drugs_max_len]
    approved_drugs_set = set(approved_drugs_list)
    print('Final number of drugs (after length removal): ' + str(len(approved_drugs_set)))
    print(' ')

    # testset loader
    approved_drugs_loader = get_dataloader(model_in, args, approved_drugs_list, batch_size=args.batch_size, shuffle=False)


    all_drug_matches = dict()
    num_valid_molecule_matches = 0
    unique_drugs_found = set()
    unique_valid_molecules_found = set()
    for i, input_batch in enumerate(approved_drugs_loader):
        current_batch_size = len(input_batch)

        # generate random seeds
        random_seed_list = get_random_list(args.approved_drugs_retries)

        # generate output batch
        output_batch = input2output_func(input_batch, random_seed_list)

        # for every molecule
        for j, input in enumerate(input_batch):
            output_unique_molecule_smiles_set = set(output_batch[j::current_batch_size])

            # num valid molecules matches
            output_valid_unique_molecule_smiles_list = \
                [output_molecule_smiles for output_molecule_smiles in output_unique_molecule_smiles_set if is_valid_molecule(output_molecule_smiles, args.property)]
            num_unique_valid_molecules_per_input = len(output_valid_unique_molecule_smiles_list)
            num_valid_molecule_matches += num_unique_valid_molecules_per_input
            unique_valid_molecules_found = unique_valid_molecules_found.union(output_valid_unique_molecule_smiles_list)

            intersection_set = approved_drugs_set.intersection(output_unique_molecule_smiles_set)
            if not not intersection_set:    # if intersection_set is not empty
                input_molecule_smiles = model_in.tensor2string(input)  # to smiles
                all_drug_matches[input_molecule_smiles] = intersection_set
                unique_drugs_found = unique_drugs_found.union(intersection_set)

            drug_index = i * current_batch_size + j
            if drug_index%30 == 0:
                print('Drug #' + str(drug_index+1) + ': Unique approved drugs found so far: ' + str(len(unique_drugs_found)))

    # print detailed results
    similarity_list = []
    out_property_list = []
    property_improvement = []
    for input_molecule_smiles, output_molecule_smiles_set in all_drug_matches.items():
        print(' ')
        property_value_in = property_calc(input_molecule_smiles, args.property)
        print('Input: ' + str(input_molecule_smiles) + '. Property value: ' + str(round(property_value_in, 4))+ '  Drug name: ' + smiles_name_dict[input_molecule_smiles])
        for output_molecule_smiles in output_molecule_smiles_set:
            property_value_out = property_calc(output_molecule_smiles, args.property)
            similarity_in_out = similarity_calc(input_molecule_smiles, output_molecule_smiles)
            print('Output: ' + str(output_molecule_smiles) + '. Property value: ' + str(round(property_value_out, 4))+ '  Drug name: ' + smiles_name_dict[output_molecule_smiles])
            print('Similarity: ' + str(round(similarity_in_out, 4)))
            similarity_list.append(similarity_in_out)
            out_property_list.append(property_value_out)
            property_improvement.append(property_value_out - property_value_in)

    # print final results
    average_drug_matches_similarity = sum(similarity_list) / len(similarity_list) if len(similarity_list)>0 else -1
    average_out_drugs_property_val = sum(out_property_list) / len(out_property_list) if len(out_property_list)>0 else -1
    average_property_improvement = sum(property_improvement) / len(property_improvement) if len(property_improvement) > 0 else -1
    unique_drugs_per_unique_legal_molecule = 100 * len(unique_drugs_found) / len(unique_valid_molecules_found) if len(unique_valid_molecules_found) > 0 else -1
    drug_matches_per_valid_matches = 100 * len(out_property_list) / num_valid_molecule_matches if num_valid_molecule_matches > 0 else -1

    print(' ')
    print('Property => ' + args.property)
    print('Translation direction   => ' + args.test_direction)

    print('Unique approved drugs found: ' + str(len(unique_drugs_found)))
    print('Unique legal molecules generated: ' + str(len(unique_valid_molecules_found)))
    print('Unique approved drugs per unique legal molecules generated: ' + str(round(unique_drugs_per_unique_legal_molecule, 5)) + '%')

    print('Drugs matches: ' + str(len(out_property_list)))
    print('All valid matches: ' + str(num_valid_molecule_matches))
    print('Unique drug-drug matches per drug-valid molecule matches: ' + str(round(drug_matches_per_valid_matches, 5)) + '%')

    print('Average generated drugs property value  => ' + str(round(average_out_drugs_property_val, 3)))
    print('Average drug matches fingerprint Similarity =>' + str(round(average_drug_matches_similarity, 3)))
    print('Average drug matches property value improvement  => ' + str(round(average_property_improvement, 3)))


def analyze_intermediate_outputs(src_mol, src_enc, src_trans, src_out, dst_mol, dst_enc, dst_trans, dst_out):
    # similarity between input molecules
    input_sim_list = [similarity_calc(src_mol, mol) for mol in dst_mol]

    # euclidean distance between embeddings after encoder
    enc_dis = [distance.euclidean(emb.cpu().numpy(), src_enc[0, :].cpu().numpy()) for emb in dst_enc]

    # euclidean distance between embeddings after translator
    trans_dis = [distance.euclidean(emb.cpu().numpy(), src_trans[0, :].cpu().numpy()) for emb in dst_trans]

    # similarity between output molecules
    out_sim_list = []
    for mol in dst_out:
        try:
            similarity_val = similarity_calc(src_out, mol)
            out_sim_list.append(similarity_val)
        except:
            continue

    # to array
    input_sim = np.array(input_sim_list)
    enc_dis = np.array(enc_dis)
    trans_dis = np.array(trans_dis)
    out_sim = np.array(out_sim_list)

    # get statistics
    input_sim_mean, input_sim_std = input_sim.mean(), input_sim.std()
    enc_dis_mean, enc_dis_std = enc_dis.mean(), enc_dis.std()
    trans_dis_mean, trans_dis_std = trans_dis.mean(), trans_dis.std()
    out_sim_mean, out_sim_std = out_sim.mean(), out_sim.std()

    return input_sim_mean, input_sim_std, enc_dis_mean, enc_dis_std, trans_dis_mean, trans_dis_std, out_sim_mean, out_sim_std


# intermediate analysis
def intermediate_analysis(args, model_in, input2all_func):
    assert args.use_EETN

    # get data
    data_path = 'dataset/' + args.property + '/A_train.txt'
    print(' ')
    print('Intermediate_analyse: Loading data from file   => ' + data_path)

    # build dataset from filename
    df = pd.read_csv(data_path, header=None)
    dataset_list = list(df.iloc[:, 0])

    # get random molecule
    random_seed_list = get_random_list(args.num_samples, last=len(dataset_list)-1)

    for seed in random_seed_list:
        rand_mol = dataset_list.pop(seed)

        # get intermediate for the *** RAND *** molecule
        rand_mol_loader = get_dataloader(model_in, args, [rand_mol], batch_size=1, shuffle=False)
        for i, input_batch in enumerate(rand_mol_loader):
            rand_enc, rand_trans, [rand_out_mol] = input2all_func(input_batch)

        # if source output molecule is invalid - skip to next molecule
        if not is_valid_molecule(rand_out_mol, args.property):
            continue

        sim_rand_all_sort_list = [(mol, similarity_calc(rand_mol, mol)) for mol in dataset_list]
        sim_rand_all_sort_list.sort(key=lambda x: x[1], reverse=True)
        neighbours_near_list = [mol for (mol, _) in sim_rand_all_sort_list[:args.num_neighbours]]
        neighbours_far_list = [mol for (mol, _) in sim_rand_all_sort_list[args.num_neighbours:2*args.num_neighbours]]
        # neighbours_far_list = [mol for (mol, _) in sim_rand_all_sort_list[-args.num_neighbours:]]

        # get intermediate for *** NEAR *** molecules
        near_loader = get_dataloader(model_in, args, neighbours_near_list, batch_size=args.num_neighbours, shuffle=False)
        for i, input_batch in enumerate(near_loader):
            near_enc, near_trans, near_out_mol = input2all_func(input_batch)

        # get intermediate for *** FAR *** molecules
        far_loader = get_dataloader(model_in, args, neighbours_far_list, batch_size=args.num_neighbours, shuffle=False)
        for i, input_batch in enumerate(far_loader):
            far_enc, far_trans, far_out_mol = input2all_func(input_batch)


        p = 3
        print(' ')
        print('seed   => ' + str(seed))
        # analyse *** FAR ***
        far_input_sim_mean, far_input_sim_std, far_enc_dis_mean, far_enc_dis_std, far_trans_dis_mean, far_trans_dis_std, far_out_sim_mean, far_out_sim_std =\
            analyze_intermediate_outputs(rand_mol, rand_enc, rand_trans, rand_out_mol, neighbours_far_list, far_enc, far_trans, far_out_mol)
        print('*** FAR (NOT similar) ***')
        print('Input mol similarity => mean: ' + str(round(far_input_sim_mean, p)) + '   std: ' + str(round(far_input_sim_std, p)))
        print('A_enc emb distance => mean: ' + str(round(far_enc_dis_mean, p)) + '   std: ' + str(round(far_enc_dis_std, p)))
        print('Translator emb distance => mean: ' + str(round(far_trans_dis_mean, p)) + '   std: ' + str(round(far_trans_dis_std, p)))
        print('Output emb distance => mean: ' + str(round(far_out_sim_mean, p)) + '   std: ' + str(round(far_out_sim_std, p)))

        # analyse *** NEAR ***
        near_input_sim_mean, near_input_sim_std, near_enc_dis_mean, near_enc_dis_std, near_trans_dis_mean, near_trans_dis_std, near_out_sim_mean, near_out_sim_std =\
            analyze_intermediate_outputs(rand_mol, rand_enc, rand_trans, rand_out_mol, neighbours_near_list, near_enc, near_trans, near_out_mol)
        print('*** NEAR (similar) ***')
        print('Input mol similarity => mean: ' + str(round(near_input_sim_mean, p)) + '   std: ' + str(round(near_input_sim_std, p)))
        print('A_enc emb distance => mean: ' + str(round(near_enc_dis_mean, p)) + '   std: ' + str(round(near_enc_dis_std, p)))
        print('Translator emb distance => mean: ' + str(round(near_trans_dis_mean, p)) + '   std: ' + str(round(near_trans_dis_std, p)))
        print('Output emb distance => mean: ' + str(round(near_out_sim_mean, p)) + '   std: ' + str(round(near_out_sim_std, p)))


if __name__ == "__main__":

    with torch.no_grad():
        # parse arguments
        args = parse_arguments()

        # set seed
        set_seed(args.seed)

        # set device (CPU/GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        # disable rdkit error messages
        rdkit_no_error_print()

        # load checkpoint
        if args.unified_destination is False:                       # regular
            T_AB, model_A, model_out_QED, model_out_DRD2 = load_checkpoint(args, device)
            model_out_QED.eval()
            model_out_DRD2.eval()
        else:
            T_AB, model_A, model_out = load_checkpoint_unified(args, device)
            model_out.eval()

        # evaluation mode
        T_AB.eval()
        model_A.eval()


        # initialize property value predictor
        property_init('DRD2')

        # prepare models according to translation direction
        model_in, T = (model_A, T_AB)

        # check testset
        if args.unified_destination is False:  # regular
            if args.check_testset is True:
                def test_input2output_func(input_batch, random_seed_list):
                    return input2output(args, input_batch, model_in, T, model_out_QED, random_seed_list, max_out_len=args.test_max_len)

                check_testset(args, model_in, model_out_QED, model_out_DRD2, test_input2output_func)
        else:                                   # ablation
            if args.check_testset is True:
                def test_input2output_func(input_batch, random_seed_list):
                    return input2output(args, input_batch, model_in, T, model_out, random_seed_list, max_out_len=args.test_max_len)

                check_testset(args, model_in, model_out, None, test_input2output_func)
