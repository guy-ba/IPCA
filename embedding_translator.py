import torch
import torch.nn as nn
import torch.nn.functional as F
import os



class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Translator(nn.Module):
    scalar = 100000
    def __init__(self, n_residual_blocks=4):
        super(Translator, self).__init__()

        # for fp attention
        model0 = [
                  nn.Linear(2048, 975, bias=False),
        ]
        self.model0 = nn.Sequential(*model0)

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(1, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling block
        in_features = 64
        out_features = in_features*2
        for _ in range(1):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling block
        out_features = in_features//2
        for _ in range(1):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output convolution
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, 1, 7),
                    nn.Tanh()]

        self.model = nn.Sequential(*model)

        model2 = [  nn.Linear(2304, 2304//2),
                    nn.BatchNorm1d(num_features=2304//2),
                    nn.LeakyReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(2304//2, 256)]

        self.model2 = nn.Sequential(*model2)

        model2 = [  nn.Linear(1600, 1600//2),
                    nn.BatchNorm1d(num_features=1600//2),
                    nn.LeakyReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(1600//2, 625)]
        self.model2 = nn.Sequential(*model2)

        model_no_fp = [ nn.Linear(676, 625),
                        nn.BatchNorm1d(num_features=625),
                        nn.LeakyReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(625, 625)]
        self.model_no_fp = nn.Sequential(*model_no_fp)

    def forward(self, x, fp=None):
        if fp is not None:
            w = self.model0(fp)
            fp = w

            x = torch.cat((x, fp), dim=1)
            x = self.model(x.view(x.shape[0],1,40,40))
            x = self.model2(x.view(x.shape[0], -1))
        else:       # for ablation no fp
            x = self.model(x.view(x.shape[0],1,25,25))
            x = self.model_no_fp(x.view(x.shape[0],-1))
        return x


# for ablation
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # Conv blocks
        model = [   nn.Conv2d(input_nc, 64, 3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(256, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x.view(x.shape[0],1,16,16))
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.decay_start_epoch = decay_start_epoch
        self.offset = offset
        self.n_epochs = n_epochs

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


# hold, update and print statistics
class Statistics(object):
    # Total
    loss_epoch = []

    # cycle constraints
    loss_cycle_A_epoch = []
    loss_cycle_B_epoch = []

    # KL for ablation
    loss_kl_A_epoch = []
    loss_kl_B_epoch = []

    def __init__(self):
        return

    def update(self, loss, cycle_loss_A, cycle_loss_B, kl_loss_A, kl_loss_B):
        # Total
        self.loss_epoch.append(loss)

        # cycle constraints
        self.loss_cycle_A_epoch.append(cycle_loss_A)
        self.loss_cycle_B_epoch.append(cycle_loss_B)

        # KL for ablation
        self.loss_kl_A_epoch.append(kl_loss_A)
        self.loss_kl_B_epoch.append(kl_loss_B)


    def print(self):
        # Total
        Average_loss = sum(self.loss_epoch) / len(self.loss_epoch)
        print('Average loss = ' + str(Average_loss.tolist()))

        # cycle constraints
        if any(self.loss_cycle_A_epoch):
            Average_loss_cycle_A = sum(self.loss_cycle_A_epoch) / len(self.loss_cycle_A_epoch)
            print('Average loss_cycle_A = ' + str(Average_loss_cycle_A.tolist()))
        if any(self.loss_cycle_B_epoch):
            Average_loss_cycle_B = sum(self.loss_cycle_B_epoch) / len(self.loss_cycle_B_epoch)
            print('Average loss_cycle_B = ' + str(Average_loss_cycle_B.tolist()))

        # KL for ablation
        if any(self.loss_kl_A_epoch):
            Average_loss_kl_A = sum(self.loss_kl_A_epoch) / len(self.loss_kl_A_epoch)
            print('Average loss_kl_A = ' + str(Average_loss_kl_A.tolist()))
        if any(self.loss_kl_B_epoch):
            Average_loss_kl_B = sum(self.loss_kl_B_epoch) / len(self.loss_kl_B_epoch)
            print('Average loss_kl_B = ' + str(Average_loss_kl_B.tolist()))


def save_checkpoint(current_criterion, best_criterion, T_AB, model_A, model_B_QED, model_B_DRD2, args):
    # first model or best model so far
    if best_criterion is None or current_criterion > best_criterion:
        best_criterion = current_criterion
        saved_state = dict(T_AB=T_AB.state_dict(),
                           model_A=model_A,
                           model_B_QED=model_B_QED,
                           model_B_DRD2=model_B_DRD2)
        checkpoint_filename_path = args.checkpoints_folder + '/' + args.property + '/checkpoint_model.pth'
        torch.save(saved_state, checkpoint_filename_path)
        print('*** Saved checkpoint in: ' + checkpoint_filename_path + ' ***')
    return best_criterion


def load_checkpoint(args, device):
    checkpoint_filename_path = args.checkpoints_folder + '/' + args.property + '/checkpoint_model.pth'
    if os.path.isfile(checkpoint_filename_path):
        print('*** Loading checkpoint file ' + checkpoint_filename_path)
        saved_state = torch.load(checkpoint_filename_path, map_location=device)

        # create embedding translator
        T_AB = Translator().to(device)

        T_AB.load_state_dict(saved_state['T_AB'])
        model_A = saved_state['model_A']
        model_B_QED = saved_state['model_B_QED']
        model_B_DRD2 = saved_state['model_B_DRD2']
    return T_AB, model_A, model_B_QED, model_B_DRD2




########################################################
def save_checkpoint_unified(current_criterion, best_criterion, T_AB, model_A, model_B, args):
    # first model or best model so far
    if best_criterion is None or current_criterion > best_criterion:
        best_criterion = current_criterion
        saved_state = dict(T_AB=T_AB.state_dict(),
                           model_A=model_A,
                           model_B=model_B)
        checkpoint_filename_path = args.checkpoints_folder + '/' + args.property + '/checkpoint_model_unified.pth'
        torch.save(saved_state, checkpoint_filename_path)
        print('*** Saved checkpoint in: ' + checkpoint_filename_path + ' ***')
    return best_criterion


def load_checkpoint_unified(args, device):
    checkpoint_filename_path = args.checkpoints_folder + '/' + args.property + '/checkpoint_model_unified.pth'
    if os.path.isfile(checkpoint_filename_path):
        print('*** Loading checkpoint file ' + checkpoint_filename_path)
        saved_state = torch.load(checkpoint_filename_path, map_location=device)

        # create embedding translator
        T_AB = Translator().to(device)

        T_AB.load_state_dict(saved_state['T_AB'])
        model_A = saved_state['model_A']
        model_B = saved_state['model_B']
    return T_AB, model_A, model_B
########################################################