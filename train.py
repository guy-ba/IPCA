import argparse
import torch
import itertools
import matplotlib.pyplot as plt

# my files
from data_preprocess import create_dataset, Dataset
from embedder import Embedder
from embedder_train import fit, get_model_train_params, get_dataloader
from embedding_translator import Translator, weights_init_normal, LambdaLR, Statistics, save_checkpoint, save_checkpoint_unified
from property_handler import smiles2fingerprint, rdkit_no_error_print
from validation import general_validation
from common_utils import set_seed, input2output, get_random_list
from decoder import Decoder

high_DRD2_loss_coef = 1.0
high_QED_loss_coef = 1.0


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='IPCA'
    )

    # end-end model settings
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train the end-end model')
    parser.add_argument('--epoch_init', type=int, default=1, help='initial epoch')
    parser.add_argument('--epoch_decay', type=int, default=120, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for end-end model')
    parser.add_argument('--property', type=str, default='M', help='name of property to translate (should be folder with that name inside dataset)')
    parser.add_argument('--init_lr', type=float, default=0.00015, help='initial learning rate')
    parser.add_argument('--is_valid', default=True, action='store_true', help='run validation every train epoch')
    parser.add_argument('--valid_direction', type=str, default='AB', help='direction of validation translation- AB: A->B; BA: B->A')
    parser.add_argument('--plot_results', default=True, action='store_true', help='plot validation set results during end-end model training')
    parser.add_argument('--print_results', default=True, action='store_true', help='print validation results during end-end model training')
    parser.add_argument('--rebuild_dataset', default=False, action='store_false', help='rebuild dataset files')
    parser.add_argument('--checkpoints_folder', type=str, default='checkpoints', help='name of folder for checkpoints saving')
    parser.add_argument('--plots_folder', type=str, default='plots_output', help='name of folder for plots saving')
    parser.add_argument('--early_stopping', type=int, default=15, help='Whether to stop training early if there is no\
                        criterion improvement for this number of validation runs.')
    parser.add_argument('--seed', type=int, default=50, help='base seed')
    parser.add_argument('--num_retries', type=int, default=20, help='number of retries for each validation sample')
    parser.add_argument('--SR_similarity', type=int, default=0.3, help='minimal similarity for success')
    parser.add_argument('--SR_property_QED_val', type=int, default=0.7, help='minimal property value for success')
    parser.add_argument('--SR_property_DRD2_val', type=int, default=0.3, help='minimal property value for success')
    parser.add_argument('--validation_max_len', type=int, default=90, help='length of validation smiles')
    parser.add_argument('--validation_batch_size', type=int, default=32, help='batch size for validation end-end model')
    parser.add_argument('--validation_freq', type=int, default=3, help='validate every n-th epoch')
    parser.add_argument('--cycle_loss', default=True, action='store_true', help='use cycle loss during training or not')

    # Ablation (old)
    parser.add_argument('--kl_loss', default=False, action='store_true', help='use kl loss during training or not')
    parser.add_argument('--use_fp', default=True, action='store_true', help='does translator use molecule fp')

    # Ablation (new)
    parser.add_argument('--unified_destination', default=False, action='store_true', help='uses 1 B domain for all molecules')
    parser.add_argument('--fixed_loss_coef', default=False, action='store_true', help='loss coefficients do not change during training')

    args = parser.parse_args()
    return args


def train_iteration_T(real_A, real_B_QED, real_B_DRD2, model_A, model_B_QED, model_B_DRD2, T_AB, T_B_QED_A, T_B_DRD2_A, optimizer_T, args):
    # prepare fingerprints
    if args.use_fp:
        real_A_fp_str = [smiles2fingerprint(model_A.tensor2string(mol), fp_translator=True) for mol in real_A]
        real_B_QED_fp_str = [smiles2fingerprint(model_B_QED.tensor2string(mol), fp_translator=True) for mol in
                             real_B_QED]
        real_B_DRD2_fp_str = [smiles2fingerprint(model_B_DRD2.tensor2string(mol), fp_translator=True) for mol in
                              real_B_DRD2]
        real_A_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in real_A_fp_str]).to(device)
        real_B_QED_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in real_B_QED_fp_str]).to(device)
        real_B_DRD2_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in real_B_DRD2_fp_str]).to(device)
        real_A_fp = real_A_fp.detach()
        real_B_QED_fp = real_B_QED_fp.detach()
        real_B_DRD2_fp = real_B_DRD2_fp.detach()
    else:
        real_A_fp, real_B_QED_fp, real_B_DRD2_fp = None, None, None


    A_times = 1
    B_QED_times = 1
    B_DRD2_times = 1
    while A_times + B_DRD2_times + B_QED_times > 0:
        optimizer_T.zero_grad()

        # embedder (METN) + translator (EETN) + cycle
        if A_times > 0:
            real_A_emb, kl_loss_A = model_A.forward_encoder(real_A)
            fake_B_emb = T_AB(real_A_emb, real_A_fp)

            # Cycle loss
            cycle_A_QED_emb = T_B_QED_A(fake_B_emb, real_A_fp)
            cycle_A_DRD2_emb = T_B_DRD2_A(fake_B_emb, real_A_fp)
            cycle_loss_A_QED, _ = model_A.forward_decoder(real_A, cycle_A_QED_emb)
            cycle_loss_A_DRD2, _ = model_A.forward_decoder(real_A, cycle_A_DRD2_emb)

        if B_QED_times > 0:
            real_B_QED_emb, kl_loss_B_QED = model_B_QED.forward_encoder(real_B_QED)
            fake_A_QED_emb = T_B_QED_A(real_B_QED_emb, real_B_QED_fp)

            # Cycle loss
            cycle_B_QED_emb = T_AB(fake_A_QED_emb, real_B_QED_fp)
            cycle_loss_B_QED, _ = model_B_QED.forward_decoder(real_B_QED, cycle_B_QED_emb)

        if B_DRD2_times > 0:
            real_B_DRD2_emb, kl_loss_B_DRD2 = model_B_DRD2.forward_encoder(real_B_DRD2)
            fake_A_DRD2_emb = T_B_DRD2_A(real_B_DRD2_emb, real_B_DRD2_fp)

            # Cycle loss
            cycle_B_DRD2_emb = T_AB(fake_A_DRD2_emb, real_B_DRD2_fp)
            cycle_loss_B_DRD2, _ = model_B_DRD2.forward_decoder(real_B_DRD2, cycle_B_DRD2_emb)

        if args.kl_loss is False:
            kl_loss_A, kl_loss_B = None, None

        # Total loss
        if args.kl_loss is False:    # Main model: only cycle
            if A_times > 0 and B_QED_times > 0 and B_DRD2_times > 0:
                loss = (cycle_loss_A_QED + cycle_loss_A_DRD2) + high_QED_loss_coef*cycle_loss_B_QED + high_DRD2_loss_coef*cycle_loss_B_DRD2
            elif A_times == 0 and B_QED_times == 0 and B_DRD2_times > 0:
                loss = high_DRD2_loss_coef*cycle_loss_B_DRD2
        else:
            print('No such setting for the main model, nor for the ablation tests')
            exit()

        loss.backward()
        optimizer_T.step()

        A_times = A_times if A_times == 0 else A_times - 1
        B_QED_times = B_QED_times if B_QED_times == 0 else B_QED_times - 1
        B_DRD2_times = B_DRD2_times if B_DRD2_times == 0 else B_DRD2_times - 1

    return loss, cycle_loss_A_QED, cycle_loss_A_DRD2, cycle_loss_B_QED, cycle_loss_B_DRD2, kl_loss_A, kl_loss_B


#########################################################
def train_iteration_T_unified(real_A, real_B, model_A, model_B, T_AB, T_BA, optimizer_T, args):
    # prepare fingerprints
    if args.use_fp:
        real_A_fp_str = [smiles2fingerprint(model_A.tensor2string(mol), fp_translator=True) for mol in real_A]
        real_B_fp_str = [smiles2fingerprint(model_B.tensor2string(mol), fp_translator=True) for mol in real_B]
        real_A_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in real_A_fp_str]).to(device)
        real_B_fp = torch.tensor([[float(dig) for dig in fp_mol] for fp_mol in real_B_fp_str]).to(device)
        real_A_fp = real_A_fp.detach()
        real_B_fp = real_B_fp.detach()
    else:
        real_A_fp, real_B_fp = None, None


    A_times = 1
    B_times = 1
    while A_times + B_times > 0:
        optimizer_T.zero_grad()

        # embedder (METN) + translator (EETN) + cycle
        if A_times > 0:
            real_A_emb, kl_loss_A = model_A.forward_encoder(real_A)
            fake_B_emb = T_AB(real_A_emb, real_A_fp)

            # Cycle loss
            cycle_A_emb = T_BA(fake_B_emb, real_A_fp)
            cycle_loss_A, _ = model_A.forward_decoder(real_A, cycle_A_emb)

        if B_times > 0:
            real_B_emb, kl_loss_B = model_B.forward_encoder(real_B)
            fake_A_emb = T_BA(real_B_emb, real_B_fp)

            # Cycle loss
            cycle_B_emb = T_AB(fake_A_emb, real_B_fp)
            cycle_loss_B, _ = model_B.forward_decoder(real_B, cycle_B_emb)

        if args.kl_loss is False:
            kl_loss_A, kl_loss_B = None, None

        # Total loss
        if args.kl_loss is False:    # Main model: only cycle
            loss = cycle_loss_A + ((high_QED_loss_coef+high_DRD2_loss_coef)/2)*cycle_loss_B
        else:
            print('No such setting for the main model, nor for the ablation tests')
            exit()

        loss.backward()
        optimizer_T.step()

        A_times = A_times if A_times == 0 else A_times - 1
        B_times = B_times if B_times == 0 else B_times - 1

    return loss, cycle_loss_A, cycle_loss_B, kl_loss_A, kl_loss_B



def validation_unified(args, model_name, model_in, model_B, T, epoch, boundaries, random_seed_list, fig=None, ax=None):
    # evaluation mode
    model_in.eval()
    model_B.eval()
    T.eval()

    # dataset loader
    valid_loader = get_dataloader(model_in, args, model_in.dataset.validset, batch_size=args.validation_batch_size, shuffle=False)

    # number samples in validset
    validset_len = len(model_in.dataset.validset)

    # tensor to molecule smiles
    def input_tensor2string(input_tensor):
        return model_in.tensor2string(input_tensor)

    trainset = set(model_in.dataset.trainset).union(set(model_B.dataset.trainset))

    #generate output molecule from input molecule
    def local_input2output(input_batch):
        return input2output(args, input_batch, model_in, T, model_B, random_seed_list, max_out_len=args.validation_max_len)

    # use general validation function
    avg_similarity, avg_property_DRD2, avg_property_QED, avg_SR, avg_validity, avg_novelty, avg_diversity =\
        general_validation(args, local_input2output, input_tensor2string, boundaries, valid_loader, validset_len, model_name, trainset, epoch,
        fig=fig, ax=ax)


    default = 30
    global high_DRD2_loss_coef, high_QED_loss_coef
    high_DRD2_loss_coef = 3*default
    high_QED_loss_coef = default
    print('high_DRD2_loss_coef = ' + str(high_DRD2_loss_coef))
    print('high_QED_loss_coef = ' + str(high_QED_loss_coef))

    # back to train mode
    model_in.train()
    model_B.train()
    T.train()

    return avg_similarity, avg_property_DRD2, avg_property_QED, avg_SR, avg_validity, avg_novelty, avg_diversity
#########################################################



def validation(args, model_name, model_in, model_B_QED, model_B_DRD2, T, epoch, boundaries, random_seed_list, fig=None, ax=None):
    # evaluation mode
    model_in.eval()
    model_B_QED.eval()
    model_B_DRD2.eval()
    T.eval()

    # dataset loader
    valid_loader = get_dataloader(model_in, args, model_in.dataset.validset, batch_size=args.validation_batch_size, shuffle=False)

    # number samples in validset
    validset_len = len(model_in.dataset.validset)

    # tensor to molecule smiles
    def input_tensor2string(input_tensor):
        return model_in.tensor2string(input_tensor)

    trainset = set(model_in.dataset.trainset).union(set(model_B_QED.dataset.trainset), set(model_B_DRD2.dataset.trainset))

    # generate output molecule from input molecule
    def local_input2output(input_batch):
        return input2output(args, input_batch, model_in, T, model_B_QED, random_seed_list, max_out_len=args.validation_max_len)

    # use general validation function
    avg_similarity, avg_property_DRD2, avg_property_QED, avg_SR, avg_validity, avg_novelty, avg_diversity =\
        general_validation(args, local_input2output, input_tensor2string, boundaries, valid_loader, validset_len, model_name, trainset, epoch,
        fig=fig, ax=ax)


    default = 30
    global high_DRD2_loss_coef, high_QED_loss_coef
    if args.fixed_loss_coef is False:       # regular
        high_DRD2_loss_coef = 3*default*((args.SR_property_DRD2_val / avg_property_DRD2)**1) if avg_property_DRD2 > 0 else 3*default
        high_QED_loss_coef = default*((args.SR_property_QED_val / avg_property_QED)**1) if avg_property_QED > 0 else default
    else:                                   # ablation
        high_DRD2_loss_coef = 3*default
        high_QED_loss_coef = default
    print('high_DRD2_loss_coef = ' + str(high_DRD2_loss_coef))
    print('high_QED_loss_coef = ' + str(high_QED_loss_coef))

    # back to train mode
    model_in.train()
    model_B_QED.train()
    model_B_DRD2.train()
    T.train()

    return avg_similarity, avg_property_DRD2, avg_property_QED, avg_SR, avg_validity, avg_novelty, avg_diversity

def early_stop(early_stopping, current_criterion, best_criterion, runs_without_improvement):
    if early_stopping is not None:
        # first model or best model so far
        if best_criterion is None or current_criterion > best_criterion:
            runs_without_improvement = 0
        # no improvement
        else:
            runs_without_improvement += 1
        if runs_without_improvement >= early_stopping:
            return True, runs_without_improvement       # True = stop training
        else:
            return False, runs_without_improvement


if __name__ == "__main__":

    # parse arguments
    args = parse_arguments()

    # set seed
    set_seed(args.seed)

    # set device (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # disable rdkit error messages
    rdkit_no_error_print()

    # epochs for METN pre training
    embedder_epochs_num = 5

    # prepare dataset
    dataset_file_A, dataset_file_B, boundaries = create_dataset(args)
    if args.unified_destination is False:           # regular
        dataset_A = Dataset(dataset_file_A, isB=False)
        dataset_B_QED = Dataset(dataset_file_B+'_QED', isB=True)
        dataset_B_DRD2 = Dataset(dataset_file_B + '_DRD2', isB=True)

        data = dataset_B_QED.trainset + dataset_B_DRD2.trainset
        chars = set()
        for string in data:
            chars.update(string)
        all_sys = sorted(list(chars)) + ['<bos>', '<eos>', '<pad>', '<unk>']
        dataset_B_QED.vocab, dataset_B_DRD2.vocab = all_sys, all_sys
        dataset_B_QED.c2i, dataset_B_DRD2.c2i = {c: i for i, c in enumerate(all_sys)}, {c: i for i, c in enumerate(all_sys)}
        dataset_B_QED.i2c, dataset_B_DRD2.i2c = {i: c for i, c in enumerate(all_sys)}, {i: c for i, c in enumerate(all_sys)}

        # create and pre-train the embedders (METNs)
        decoder_A = Decoder(len(dataset_A.vocab), dataset_A.c2i)
        model_A = Embedder(dataset_A, 'Embedder A', decoder_A).to(device)
        fit(args, model_A, 4*embedder_epochs_num, boundaries, is_validation=True)

        # decoder for all B sets
        decoder_shared_B = Decoder(len(dataset_B_QED.vocab), dataset_B_QED.c2i)

        model_B_QED = Embedder(dataset_B_QED, 'Embedder B QED', decoder_shared_B).to(device)
        fit(args, model_B_QED, embedder_epochs_num, boundaries, is_validation=False)

        model_B_DRD2 = Embedder(dataset_B_DRD2, 'Embedder B DRD2', decoder_shared_B).to(device)
        fit(args, model_B_DRD2, 5*embedder_epochs_num, boundaries, is_validation=False)
    else:               # unified ablation
        dataset_A = Dataset(dataset_file_A+'_unified', isB=False)
        dataset_B = Dataset(dataset_file_B+'_unified', isB=True)

        # create  and pre-train the embedders (METNs)
        decoder_A = Decoder(len(dataset_A.vocab), dataset_A.c2i)
        model_A = Embedder(dataset_A, 'Embedder A', decoder_A).to(device)
        fit(args, model_A, 4*embedder_epochs_num, boundaries, is_validation=True)

        # decoder for B set
        decoder_shared_B = Decoder(len(dataset_B.vocab), dataset_B.c2i)
        model_B = Embedder(dataset_B, 'Embedder B', decoder_shared_B).to(device)
        fit(args, model_B, 4*embedder_epochs_num, boundaries, is_validation=False)

    # create embedding translators (EETN) and weights
    T_AB = Translator().to(device)
    T_AB.apply(weights_init_normal)
    if args.unified_destination is False:           # regular
        T_B_QED_A = Translator().to(device)
        T_B_DRD2_A = Translator().to(device)
        T_B_QED_A.apply(weights_init_normal)
        T_B_DRD2_A.apply(weights_init_normal)
    else:                                           # unified ablation
        T_BA = Translator().to(device)
        T_BA.apply(weights_init_normal)

    # optimizer
    if args.unified_destination is False:           # regular
        optimizer_T = torch.optim.Adam(itertools.chain(T_AB.parameters(), T_B_QED_A.parameters(), T_B_DRD2_A.parameters(), get_model_train_params(model_A),
                                    get_model_train_params(model_B_QED), get_model_train_params(model_B_DRD2), decoder_A.parameters(),
                                                   decoder_shared_B.parameters()), lr=args.init_lr, betas=(0.5, 0.999))
    else:                                           # unified ablation
        optimizer_T = torch.optim.Adam(itertools.chain(T_AB.parameters(), T_BA.parameters(), get_model_train_params(model_A),
                                    get_model_train_params(model_B), decoder_A.parameters(),
                                                   decoder_shared_B.parameters()), lr=args.init_lr, betas=(0.5, 0.999))


    # scheduler
    lr_scheduler_T = torch.optim.lr_scheduler.LambdaLR(optimizer_T, lr_lambda=LambdaLR(args.epochs, args.epoch_init, args.epoch_decay).step)


    # train dataloaders
    A_train_loader = get_dataloader(model_A, args, model_A.dataset.trainset, args.batch_size, collate_fn=None, shuffle=True)
    if args.unified_destination is False:  # regular
        B_QED_train_loader = get_dataloader(model_B_QED, args, model_B_QED.dataset.trainset, args.batch_size, collate_fn=None, shuffle=True)
        B_DRD2_train_loader = get_dataloader(model_B_DRD2, args, model_B_DRD2.dataset.trainset, args.batch_size,
                                            collate_fn=None, shuffle=True)
    else:
        B_train_loader = get_dataloader(model_B, args, model_B.dataset.trainset, args.batch_size, collate_fn=None, shuffle=True)

    # for early stopping
    best_criterion = None
    runs_without_improvement = 0

    # generate random seeds
    random_seed_list = get_random_list(args.num_retries)

    ###### Training ######
    for epoch in range(args.epoch_init, args.epochs + 1):
        print(' ')
        print('epoch #' + str(epoch))

        # statistics
        stats = Statistics()
        if args.unified_destination is False:                       # regular
            for i, (real_A, real_B_QED, real_B_DRD2) in enumerate(zip(A_train_loader, B_QED_train_loader, B_DRD2_train_loader)):

                # update translators (EETN) and embedders (METNs)
                loss, cycle_loss_A_QED, cycle_loss_A_DRD2, cycle_loss_B_QED, cycle_loss_B_DRD2, kl_loss_A, kl_loss_B = \
                train_iteration_T(real_A, real_B_QED, real_B_DRD2, model_A, model_B_QED, model_B_DRD2, T_AB, T_B_QED_A, T_B_DRD2_A, optimizer_T, args)


                # update statistics
                cycle_loss_A = cycle_loss_A_QED + cycle_loss_A_DRD2
                cycle_loss_B = cycle_loss_B_QED + cycle_loss_B_DRD2
                stats.update(loss, cycle_loss_A, cycle_loss_B, kl_loss_A, kl_loss_B)

            # print epoch's statistics
            stats.print()

            # run validation
            if args.is_valid is True and (epoch == 1 or epoch % args.validation_freq == 0):
                if args.valid_direction is 'AB' or args.valid_direction is 'Both':
                    if epoch == 1:
                        fig_AB, ax_AB = plt.subplots()
                    avg_similarity_AB, avg_property_DRD2_AB, avg_property_QED_AB, avg_SR_AB, avg_validity_AB, avg_novelty_AB, avg_diversity_AB = \
                        validation(args, 'Our AB', model_A, model_B_QED, model_B_DRD2, T_AB, epoch, boundaries, random_seed_list, fig=fig_AB, ax=ax_AB)
                    # save plots
                    if args.plot_results is True:
                        fig_AB.savefig(args.plots_folder + '/' + args.property + '/Our AB valid')


                # early stopping
                avg_property_AB = avg_property_DRD2_AB + avg_property_QED_AB
                current_criterion = 10*avg_SR_AB + avg_property_AB + avg_similarity_AB + avg_validity_AB
                is_early_stop, runs_without_improvement = \
                    early_stop(args.early_stopping, current_criterion, best_criterion, runs_without_improvement)
                if is_early_stop:
                    break

                # save checkpoint
                best_criterion = save_checkpoint(current_criterion, best_criterion, T_AB, model_A, model_B_QED, model_B_DRD2, args)


        else:                       # ablation
            for i, (real_A, real_B) in enumerate(zip(A_train_loader, B_train_loader)):
                # update translators (EETN) and embedders (METNs)
                loss, cycle_loss_A, cycle_loss_B, kl_loss_A, kl_loss_B = \
                    train_iteration_T_unified(real_A, real_B, model_A, model_B, T_AB, T_BA, optimizer_T, args)

                # update statistics
                stats.update(loss, cycle_loss_A, cycle_loss_B, kl_loss_A, kl_loss_B)

            # print epoch's statistics
            stats.print()

            # run validation
            if args.is_valid is True and (epoch == 1 or epoch % args.validation_freq == 0):
                if args.valid_direction is 'AB' or args.valid_direction is 'Both':
                    if epoch == 1:
                        fig_AB, ax_AB = plt.subplots()
                    avg_similarity_AB, avg_property_DRD2_AB, avg_property_QED_AB, avg_SR_AB, avg_validity_AB, avg_novelty_AB, avg_diversity_AB = \
                        validation_unified(args, 'Our AB unified', model_A, model_B, T_AB, epoch, boundaries,
                                   random_seed_list, fig=fig_AB, ax=ax_AB)
                    # save plots
                    if args.plot_results is True:
                        fig_AB.savefig(args.plots_folder + '/' + args.property + '/Our AB valid unified')

                # early stopping
                avg_property_AB = avg_property_DRD2_AB + avg_property_QED_AB
                current_criterion = 10 * avg_SR_AB + avg_property_AB + avg_similarity_AB + avg_validity_AB
                is_early_stop, runs_without_improvement = \
                    early_stop(args.early_stopping, current_criterion, best_criterion, runs_without_improvement)
                if is_early_stop:
                    break

                # save checkpoint
                best_criterion = save_checkpoint_unified(current_criterion, best_criterion, T_AB, model_A, model_B, args)

        # update learning rate
        lr_scheduler_T.step()
