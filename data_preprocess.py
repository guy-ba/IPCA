import pandas as pd
import torch
import numpy as np
from collections import OrderedDict

# my files
from property_handler import property_init, property_calc, canonicalize_smiles

# property - DRUG LIKENESS (QED)
QED_dataset_path = 'dataset/QED/'
QED_dataset_filename = 'QED_DATASET.txt'
QED_high_A = 0.78
QED_low_B = 0.92

# property - DRD2
DRD2_dataset_path = 'dataset/DRD2/'
DRD2_dataset_filename = 'DRD2_DATASET.txt'
DRD2_high_A = 0.05
DRD2_low_B = 0.65

# property - M
dataset_M_path = 'dataset/M/'
M_valid_from_train = 0
M_test = 780



def create_dataset(args):
    if args.rebuild_dataset is True:
        property_init('DRD2')

        ###### VALIDATION and TEST ######
        test_filename = 'A_test.txt'

        # QED - extract both low prop
        test_cand_QED = set(get_unpaired_datasets(QED_dataset_path + test_filename))
        test_QED = []
        for i, mol_smiles in enumerate(test_cand_QED):
            property_val_QED = property_calc(mol_smiles, 'QED')
            property_val_DRD2 = property_calc(mol_smiles, 'DRD2')
            # these threshold for validation and test molecules are taken from the HG2G paper [8]
            if property_val_QED <= 0.8 and property_val_DRD2 < 0.05:
                test_QED.append(mol_smiles)

        # DRD2 - extract both low prop
        test_cand_DRD2 = set(get_unpaired_datasets(DRD2_dataset_path + test_filename)) - set(test_QED)
        test_DRD2 = []
        for i, mol_smiles in enumerate(test_cand_DRD2):
            property_val_QED = property_calc(mol_smiles, 'QED')
            property_val_DRD2 = property_calc(mol_smiles, 'DRD2')
            # these threshold for validation and test molecules are taken from the HG2G paper [8]
            if property_val_QED <= 0.8 and property_val_DRD2 < 0.05:
                test_DRD2.append(mol_smiles)


        ###### TRAIN ######
        # DRD2 - extract low DRD2
        no_dup_DRD2 = set(get_unpaired_datasets(DRD2_dataset_path + DRD2_dataset_filename)) - set(test_QED+test_DRD2)
        low_DRD2, high_DRD2 = [], []
        for i, mol_smiles in enumerate(no_dup_DRD2):
            drd2_val = property_calc(mol_smiles, 'DRD2')

            ################# remove all molecules with properties above thresholds #################
            qed_val = property_calc(mol_smiles, 'QED')
            if drd2_val>=args.SR_property_DRD2_val and qed_val>=args.SR_property_QED_val:
                continue
            ##################################
            if drd2_val >= DRD2_low_B:
                high_DRD2.append(mol_smiles)
            elif drd2_val <= DRD2_high_A:
                low_DRD2.append(mol_smiles)

        # QED - extract low QED
        no_dup_QED = set(get_unpaired_datasets(QED_dataset_path + QED_dataset_filename)) - set(test_QED+test_DRD2+low_DRD2+high_DRD2)
        low_QED, high_QED = [], []
        for i, mol_smiles in enumerate(no_dup_QED):
            qed_val = property_calc(mol_smiles, 'QED')

            ################# remove all molecules with properties above thresholds #################
            drd2_val = property_calc(mol_smiles, 'DRD2')
            if drd2_val>=args.SR_property_DRD2_val and qed_val>=args.SR_property_QED_val:
                continue
            ##################################

            if qed_val >= QED_low_B:
                high_QED.append(mol_smiles)
            elif qed_val <= QED_high_A:
                low_QED.append(mol_smiles)


        # calc and print lens
        len_test_QED, len_test_DRD2 = len(test_QED), len(test_DRD2)
        len_low_DRD2, len_high_DRD2, len_low_QED, len_high_QED = len(low_DRD2), len(high_DRD2), len(low_QED), len(high_QED)
        print("len_test_QED = " + str(len_test_QED))
        print("len_test_DRD2 = " + str(len_test_DRD2))
        print("len_low_DRD2 = " + str(len_low_DRD2))
        print("len_high_DRD2 = " + str(len_high_DRD2))
        print("len_low_QED = " + str(len_low_QED))
        print("len_high_QED = " + str(len_high_QED))

        ###### TRAIN ######
        # A train (low property)
        # regular
        A_train = np.random.choice(a=low_DRD2, size=len_high_DRD2//2, replace=False).tolist()
        A_train.extend(np.random.choice(a=low_QED, size=len_high_DRD2 - len(A_train), replace=False).tolist())
        assert len(A_train) == len_high_DRD2
        final_A_train, _, _ = create_dataset_files(A_train, dataset_M_path + 'A', len(A_train), 0, 0)

        # ablation
        A_train_unified = np.random.choice(a=low_DRD2, size=len_high_DRD2, replace=False).tolist()
        A_train_unified.extend(np.random.choice(a=low_QED, size=len_high_DRD2, replace=False).tolist())
        assert len(A_train_unified) == 2*len_high_DRD2
        final_A_train_unified, _, _ = create_dataset_files(A_train_unified, dataset_M_path + 'A_unified', len(A_train_unified), 0, 0)

        # B train (high property)
        # regular
        final_B_QED_train, _, _ = create_dataset_files(high_QED, dataset_M_path + 'B_' + 'QED', len_high_DRD2, 0, 0)
        final_B_DRD2_train, _, _ = create_dataset_files(high_DRD2, dataset_M_path + 'B_' + 'DRD2', len_high_DRD2, 0, 0)

        # ablation
        unified_B = final_B_QED_train+final_B_DRD2_train
        final_B_DRD2_train_unified, _, _ = create_dataset_files(unified_B, dataset_M_path + 'B_unified', len(unified_B), 0, 0)


    # declare domains boundaries
    boundaries = Boundary(QED_low_B, DRD2_low_B)
    return dataset_M_path + 'A', dataset_M_path + 'B', boundaries



def get_unpaired_datasets(path):
    data = pd.read_csv(path, header=None)
    paired_list = (data.squeeze()).astype(str).tolist()
    upaired_list = [smiles for pair in paired_list for smiles in pair.split()]
    upaired_list_no_dup = list(OrderedDict.fromkeys(upaired_list))
    return upaired_list_no_dup


def create_dataset_files(data, out_file_path, trainset_size, validset_size=None, testset_size=None):
    if trainset_size + validset_size + testset_size > 0:
        train_valid_test_set = np.random.choice(a=data, size=trainset_size + validset_size + testset_size, replace=False).tolist()
    else:
        train_valid_test_set = []

    # train
    if trainset_size > 0:
        train_set = train_valid_test_set[:trainset_size]
        pd.DataFrame(train_set).to_csv(out_file_path + '_train.txt', header=False, index=False)
    else:
        train_set = None

    # validation
    if validset_size > 0:
        valid_set = train_valid_test_set[trainset_size:trainset_size+validset_size]
        pd.DataFrame(valid_set).to_csv(out_file_path + '_validation.txt', header=False, index=False)
    else:
        valid_set = None

    # test
    if testset_size > 0:
        test_set = train_valid_test_set[trainset_size+validset_size:trainset_size+validset_size+testset_size]
        pd.DataFrame(test_set).to_csv(out_file_path + '_test.txt', header=False, index=False)
    else:
        test_set = None

    return train_set, valid_set, test_set


# for holding domains boundaries
class Boundary(object):
    A_boundary = None
    B_boundary = None
    middle = None
    def __init__(self, A_boundary, B_boundary):
        self.A_boundary = A_boundary
        self.B_boundary = B_boundary
        self.middle = (A_boundary + B_boundary)/2
    def get_boundary(self, domain):
        if domain is 'A':
            return self.A_boundary
        elif domain is 'B':
            return self.B_boundary
        else:
            return self.middle


# for holding datasets, 1 for A and 1 for B
class Dataset(object):
    trainset = None
    validset = None
    vocab = None
    c2i = None
    i2c = None

    def __init__(self, filename, isB=False):
        self.trainset = pd.read_csv(filename + '_train.txt', header=None).iloc[:,0].tolist()
        if isB:
            data = self.trainset
        else:
            self.validset = pd.read_csv(filename + '_validation.txt', header=None).iloc[:,0].tolist()
            valid_no_source = [mol.split()[0] for mol in self.validset]
            data = self.trainset + valid_no_source

        chars = set()
        for string in data:
            chars.update(string)
        all_sys = sorted(list(chars)) + ['<bos>', '<eos>', '<pad>', '<unk>']
        self.vocab = all_sys
        self.c2i = {c: i for i, c in enumerate(all_sys)}
        self.i2c = {i: c for i, c in enumerate(all_sys)}

    def in_trainset(self, string):
        return string in self.trainset


    def char2id(self, char, c2i):
        return c2i['<unk>'] if char not in c2i else c2i[char]

    def id2char(self, id, i2c):
        return i2c[32] if id not in i2c else i2c[id]

    def string2ids(self, string, c2i, add_bos=False, add_eos=False):
        ids = [self.char2id(c, c2i) for c in string]
        if add_bos:
            ids = [c2i['<bos>']] + ids
        if add_eos:
            ids = ids + [c2i['<eos>']]
        return ids
    def ids2string(self, ids, c2i, i2c, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == c2i['<bos>']:
            ids = ids[1:]
        if rem_eos and ids[-1] == c2i['<eos>']:
            ids = ids[:-1]
        string = ''.join([self.id2char(id, i2c) for id in ids])
        return string
    def string2tensor(self, string, c2i, device='model'):
        ids = self.string2ids(string, c2i, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long,device=device if device == 'model' else device)
        return tensor

def filname2testset(testset_filename, model_in, model_out_QED, model_out_DRD2, drugs=False):
    print(' ')

    # create testset loader from file
    if drugs:
        df = pd.read_csv(testset_filename)
        testset = set(df['smiles'].dropna())
    else:
        df = pd.read_csv(testset_filename, header=None)
        testset = set(df.iloc[:,0])
    print('Initial number of molecules: ' + str(len(testset)))

    canonicalized_testset = []
    for sample in testset:
        try:
            canonicalized_sample = canonicalize_smiles(sample)
            canonicalized_testset.append(canonicalized_sample)
        except:
            continue
    print('Not Nan canonicalized molecules: ' + str(len(canonicalized_testset)))

    # delete all strings containing chars not from vocab
    vocab_set = set(model_in.dataset.vocab)
    testset_restricted = [mol for mol in canonicalized_testset if set(mol).issubset(vocab_set)]
    print('With compatible vocabulary: ' + str(len(testset_restricted)))

    # delete and report overlapping molecules in testset and train+validation sets
    if model_in.dataset.validset is None:
        model_in_set = set(model_in.dataset.trainset)
    else:
        model_in_set = set(model_in.dataset.trainset).union(set(model_in.dataset.validset))
    if model_out_QED.dataset.validset is None:
        model_out_QED_set = set(model_out_QED.dataset.trainset)
    else:
        model_out_QED_set = set(model_out_QED.dataset.trainset).union(set(model_out_QED.dataset.validset))
    if model_out_DRD2.dataset.validset is None:
        model_out_DRD2_set = set(model_out_DRD2.dataset.trainset)
    else:
        model_out_DRD2_set = set(model_out_DRD2.dataset.trainset).union(set(model_out_DRD2.dataset.validset))
    model_out_set = model_out_QED_set.union(model_out_DRD2_set)

    test_train_and_valid_inter = set(testset_restricted).intersection(model_in_set.union(model_out_set))
    testset_restricted = testset_restricted if test_train_and_valid_inter == set() else list(set(testset_restricted) - test_train_and_valid_inter)
    print('After (Test) and (Train & Validation) intersection removal: ' + str(len(testset_restricted)))

    # check all characters in the testset have appeared in the train + validation sets
    chars = set()
    for string in testset_restricted:
        chars.update(string)
    test_vocab = sorted(list(chars)) + ['<bos>', '<eos>', '<pad>', '<unk>']
    testset_restricted.sort()

    if drugs:
        df_smiles_name = df.loc[df['smiles'].isin(testset_restricted)][['Name', 'smiles']]
        smiles_name_dict = df_smiles_name.set_index('smiles').to_dict()['Name']
        return testset_restricted, smiles_name_dict if set(test_vocab).issubset(vocab_set) else exit()

    return testset_restricted if set(test_vocab).issubset(vocab_set) else exit()


##########################################
def filname2testset_unified(testset_filename, model_in, model_out, drugs=False):
    print(' ')

    # create testset loader from file
    if drugs:
        df = pd.read_csv(testset_filename)
        testset = set(df['smiles'].dropna())
    else:
        df = pd.read_csv(testset_filename, header=None)
        testset = set(df.iloc[:,0])
    print('Initial number of molecules: ' + str(len(testset)))

    canonicalized_testset = []
    for sample in testset:
        try:
            canonicalized_sample = canonicalize_smiles(sample)
            canonicalized_testset.append(canonicalized_sample)
        except:
            continue
    print('Not Nan canonicalized molecules: ' + str(len(canonicalized_testset)))

    # delete all strings containing chars not from vocab
    vocab_set = set(model_in.dataset.vocab)
    testset_restricted = [mol for mol in canonicalized_testset if set(mol).issubset(vocab_set)]
    print('With compatible vocabulary: ' + str(len(testset_restricted)))

    # delete and report overlapping molecules in testset and train+validation sets
    if model_in.dataset.validset is None:
        model_in_set = set(model_in.dataset.trainset)
    else:
        model_in_set = set(model_in.dataset.trainset).union(set(model_in.dataset.validset))
    if model_out.dataset.validset is None:
        model_out_set = set(model_out.dataset.trainset)
    else:
        model_out_set = set(model_out.dataset.trainset).union(set(model_out.dataset.validset))

    test_train_and_valid_inter = set(testset_restricted).intersection(model_in_set.union(model_out_set))
    testset_restricted = testset_restricted if test_train_and_valid_inter == set() else list(set(testset_restricted) - test_train_and_valid_inter)
    print('After (Test) and (Train & Validation) intersection removal: ' + str(len(testset_restricted)))

    # check all characters in the testset have appeared in the train + validation sets
    chars = set()
    for string in testset_restricted:
        chars.update(string)
    test_vocab = sorted(list(chars)) + ['<bos>', '<eos>', '<pad>', '<unk>']
    testset_restricted.sort()

    if drugs:
        df_smiles_name = df.loc[df['smiles'].isin(testset_restricted)][['Name', 'smiles']]
        smiles_name_dict = df_smiles_name.set_index('smiles').to_dict()['Name']
        return testset_restricted, smiles_name_dict if set(test_vocab).issubset(vocab_set) else exit()

    return testset_restricted if set(test_vocab).issubset(vocab_set) else exit()

##########################################