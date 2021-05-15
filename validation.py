import torch
import matplotlib.pyplot as plt
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# my files
from property_handler import property_init
from common_utils import generate_results_file, process_results_file, valid_results_file_to_metrics


# validation function
def general_validation(args, input2output, input_tensor2string, boundaries, valid_loader, validset_len, model_name, trainset, epoch,
                        fig=None, ax=None):
    with torch.no_grad():
        if epoch == 1:
            property_init('DRD2')
            if args.plot_results is True:
                plt.ion()
                fig.suptitle(
                    args.property + ' and similarity using ' + model_name + ' as a function of the number of Epochs')
                ax.set_xlabel('Number of Epochs')
                ax.set_ylabel(args.property + ' and fingerprints similarity')
                ax.axhline(y=boundaries.get_boundary('A'))
                ax.axhline(y=boundaries.get_boundary('B'))


        results_file_path = args.plots_folder + '/' + args.property + '/IPCA_' + args.property + '_validation.txt'
        valid_results_file_path = args.plots_folder + '/' + args.property + '/valid_IPCA_' + args.property + '_validation.txt'

        # generate results file
        generate_results_file(valid_loader, input2output, input_tensor2string, results_file_path)

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
            valid_results_file_to_metrics(valid_results_file_path, args, validset_len)

        # print results
        print(' ')
        print('Property => ' + args.property)
        print('DRD2 => mean: ' + str(round(property_DRD_mean, 3)) + '   std: ' + str(round(property_DRD_std, 3)))
        print('QED => mean: ' + str(round(property_QED_mean, 3)) + '   std: ' + str(round(property_QED_std, 3)))
        print('fingerprint Similarity => mean: ' + str(round(similarity_mean, 3)) + '   std: ' + str(
            round(similarity_std, 3)))
        print('SR => mean: ' + str(round(SR_mean, 3)) + '   std: ' + str(round(SR_std, 3)))
        print('validity => mean: ' + str(round(validity_mean, 3)) + '   std: ' + str(round(validity_std, 3)))
        print('novelty => mean: ' + str(round(novelty_mean, 3)) + '   std: ' + str(round(novelty_std, 3)))
        print('diversity => mean: ' + str(round(diversity_mean, 3)) + '   std: ' + str(round(diversity_std, 3)))

        # plot
        dot_size=8
        if args.plot_results is True:
            ax.scatter(epoch, property_DRD_mean, s=dot_size, label='DRD2', c='navy')
            ax.scatter(epoch, property_QED_mean, s=dot_size, label='QED', c='deepskyblue')
            ax.scatter(epoch, similarity_mean, s=dot_size, label='similarity', c='green')
            ax.scatter(epoch, SR_mean, s=dot_size, label='SR', c='red')
            ax.scatter(epoch, validity_mean, s=dot_size, label='validity', c='olive')
            ax.scatter(epoch, novelty_mean, s=dot_size, label='novelty', c='aqua')
            ax.scatter(epoch, diversity_mean, s=dot_size, label='diversity', c='yellow')
            ax.text(epoch, 1.04*property_DRD_mean, str(round(100 * property_DRD_mean, 1)), fontsize=8, color='navy')
            ax.text(epoch, 1.04*property_QED_mean, str(round(100 * property_QED_mean, 1)), fontsize=8, color='deepskyblue')
            ax.text(epoch, 1.05*similarity_mean, str(round(100 * similarity_mean, 1)), fontsize=8, color='green')
            ax.text(epoch, 1.05*SR_mean, str(round(100 * SR_mean, 1)), fontsize=8, color='red')
            if ax.get_legend() is None:
                ax.legend()
            plt.plot()
            plt.pause(0.01)

        return similarity_mean, property_DRD_mean, property_QED_mean, SR_mean, validity_mean, novelty_mean, diversity_mean