from ConvLSTM import ConvLSTMCell
from utils import util_of_lpips, Loss
lpips_fn = util_of_lpips('vgg')
from NetworkBlock import *
import torch
import torch.nn.functional as F


class PPNv2(nn.Module):
    def __init__(self, channels, hidden_channels, merge_style='modulate'):
        super(PPNv2, self).__init__()
        self.num_layers = len(channels)
        self.channels = channels
        self.hidden_channels = hidden_channels
        for l in range(self.num_layers):
            kernel_size = (3, 3)
            rnncell = ConvLSTMCell(input_dim=channels[l], hidden_dim=self.hidden_channels[l], kernel_size=kernel_size)

            MergeError = nn.Sequential(
                nn.Conv2d(in_channels=channels[l]*2, out_channels=channels[l], kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(inplace=True)
            )

            MergeErrorInput = MergeStyle(mode=merge_style, channels=channels[l])
            MergePred = MergeStyle(mode=merge_style, channels=channels[l])

            setattr(self, 'rnncell{}'.format(l), rnncell)
            setattr(self, 'MergeErrorInput{}'.format(l), MergeErrorInput)
            setattr(self, 'MergeError{}'.format(l), MergeError)
            setattr(self, 'MergePred{}'.format(l), MergePred)

        for l in range(self.num_layers-1):
            pad_mode = 'zeros'
            UpInput = UpPropagate(in_channel=channels[l], out_channel=channels[l+1],  pad_mode=pad_mode)

            UpError = UpPropagate(in_channel=channels[l]*2, out_channel=channels[l+1], pad_mode=pad_mode)
            setattr(self, 'UpError{}'.format(l), UpError)
            DownPred = DownPropagate(in_channel=channels[l+1], out_channel=channels[l], pad_mode=pad_mode)

            setattr(self, 'UpInput{}'.format(l), UpInput)
            setattr(self, 'DownPred{}'.format(l), DownPred)

        self.error_factor = nn.Parameter(torch.ones(self.num_layers, requires_grad=True, dtype=torch.float))
        self.relu = nn.LeakyReLU(inplace=True)
        self.loss = Loss()


    def MakeErrorAndHigherInput(
            self,
            cur_pred,          # current prediction P_l^t
            cur_error,
            target,            # current target T_l^{t+1}
            next_input,        # next input f_l^{t+1} for upward propagation
            level,
            mode,
    ):
        if cur_pred == None or next_input == None:
            # no input and target for higher level and error remain unchanged
            return None, None, cur_error
        pos, neg = F.relu(cur_pred - next_input), F.relu(next_input - cur_pred)
        next_error = torch.cat([pos, neg], dim=1)

        if level >= self.num_layers - 1:
            return None, None, next_error

        # make input and target for higher level
        UpInput = getattr(self, 'UpInput{}'.format(level))
        if mode == 'train' or mode == 'adv_train':
            higher_target, higher_input = UpInput(target), UpInput(next_input)
            return higher_input, higher_target, next_error
        else:
            higher_input = UpInput(next_input)
            return higher_input, None, next_error


    def forward(self, inputs, pred_steps, mode):
        '''
        param:
            inputs.size = (b, t, c, h, w)
            pred_steps: prediction steps
            mode: 'train', 'test', 'val' or 'adv_train'
            b: batch size
            t: time step
        '''
        assert mode in ['train', 'test', 'val', 'adv_train']
        hidden_seq = []                                   # for storing hidden variable of ConvLSTM
        error_seq = [None] * self.num_layers              # for storing local errors
        list_predict = []                                 # for storing local predictions
        list_input = []                                   # for storing local inputs
        list_target = []                                  # for storing local targets
        predictions = []                                  # for storing predictions of the lowest level
        batch, time_step, c, high, width = inputs.size()  # extract the input size parameters
        mse_loss = 0
        lpips_loss = 0
        encode_mse_loss = 0
        # var = torch.var(inputs, dim=1).mean()

        if mode == 'train' or mode == 'adv_train':
            real_t = time_step - pred_steps   # the number of steps that using real frames as inputs
        else:
            real_t = time_step

        # initialization
        for l in range(self.num_layers):
            hidden_dim = self.hidden_channels[l]
            h_c = torch.zeros(batch, hidden_dim, high, width, device=inputs.device)                 # hidden or cell
            hidden_seq.append((h_c, h_c))
            error_seq[l] = torch.zeros(batch, self.channels[l] * 2, high, width, device=inputs.device)
            list_predict.append([None] * (real_t + pred_steps + 1))
            list_input.append([None] * (real_t + pred_steps + 1))
            list_target.append([None] * (real_t + pred_steps + 1))
            width = width // 2
            high = high // 2

        # load original inputs and targets
        for t in range(time_step):
            cur_input = inputs[:, t]
            if t < real_t:
                list_input[0][t] = cur_input
            if mode == 'train' or mode == 'adv_train':
                list_target[0][t] = cur_input


        for t in range(real_t + pred_steps - 1):
            for l in reversed(range(self.num_layers)):
                cur_input = list_input[l][t]
                cur_hidden = hidden_seq[l]
                if l == self.num_layers - 1:
                    # no predictions from higher level at the highest level
                    higher_pred = None
                else:
                    higher_pred = list_predict[l + 1][t]
                if l == 0:
                    # no error from lower level at the lowest level
                    lower_error = None
                else:
                    lower_error = error_seq[l - 1]
                cur_error = error_seq[l]
                #########################################################################
                # Make Prediction
                if cur_input == None:
                    list_predict[l][t] = None
                    hidden_seq[l] = cur_hidden
                else:
                    rnncell = getattr(self, 'rnncell{}'.format(l))
                    lstm_input = cur_input
                    MergeError = getattr(self, 'MergeError{}'.format(l))
                    if lower_error != None:
                        UpError = getattr(self, 'UpError{}'.format(l - 1))
                        lower_error, cur_error = UpError(lower_error), MergeError(cur_error)
                        lambda_e = torch.sigmoid(self.error_factor[l])
                        cur_error = lambda_e * cur_error + (1-lambda_e) * lower_error
                    else:
                        cur_error = MergeError(cur_error)

                    MergeErrorInput = getattr(self, 'MergeErrorInput{}'.format(l))
                    lstm_input = MergeErrorInput(lstm_input, cur_error)
                    lstm_output, next_hidden = rnncell(lstm_input, cur_hidden)

                    if higher_pred == None:
                        list_predict[l][t] = lstm_output
                        hidden_seq[l] = next_hidden
                    else:
                        MergePred = getattr(self, 'MergePred{}'.format(l))
                        DownPred = getattr(self, 'DownPred{}'.format(l))
                        higher_pred = DownPred(higher_pred)
                        list_predict[l][t] = MergePred(higher_pred, lstm_output)
                        hidden_seq[l] = next_hidden


                cur_pred = list_predict[l][t]
                if mode == 'adv_train' and l == 0:
                    predictions.append(F.relu(cur_pred))

                if t >= real_t - 1:
                    if l == 0:
                        if mode == 'test' or mode == 'val':
                            predictions.append(F.relu(cur_pred))
                        list_input[0][t+1] = F.relu(cur_pred)
                    else:
                        if (mode == 'train' or mode == 'adv_train') and t >= real_t:
                            cur_target = list_target[l][t]
                            if cur_input != None and cur_target != None:
                                lambda_t = 0.2 if t == 0 else 1.0
                                lambda_l = 1.0 / self.num_layers if l > 0 else 1.0
                                encode_mse_loss += self.loss.MSELoss(cur_input, cur_target, lambda_t, lambda_l, reduction='free')


            if  t == real_t + pred_steps - 1:
                if mode == 'test' or mode == 'val':
                    break

            for l in range(self.num_layers):
                cur_pred = list_predict[l][t]
                cur_error = error_seq[l]
                target = list_target[l][t + 1]
                next_input = list_input[l][t + 1]
                higher_input, higher_target, next_error = self.MakeErrorAndHigherInput(cur_pred, cur_error, target, next_input, l, mode)
                error_seq[l] = next_error
                if l < self.num_layers - 1:
                    list_input[l + 1][t + 1] = higher_input
                    list_target[l + 1][t + 1] = higher_target
                if (mode == 'train' or mode == 'adv_train') and cur_pred != None and target != None:
                    lambda_t = 0.2 if t == 0 else 1.0
                    lambda_l = 1.0 / self.num_layers if l > 0 else 1.0

                    mse_loss += self.loss.MSELoss(cur_pred, target, lambda_t, lambda_l, reduction='free')
                    if l == 0:
                        # using lpips loss
                        b, c, h, w = target.size()
                        lpips_loss += torch.mean(lpips_fn.calc_lpips(cur_pred, target) ** 2) * lambda_t * (c * h * w)

        if mode == 'adv_train':
            return predictions, mse_loss + lpips_loss + encode_mse_loss
        elif mode == 'train':
            return mse_loss + lpips_loss + encode_mse_loss
        else:
            return predictions



if __name__ == '__main__':
    rank = 1
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(rank)
    channels = (3, 64, 64, 256, 256, 512)
    model = PPNv2(channels, channels)
    model = model.to(0)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(10):
        x = torch.rand(1,20,3,96,128).to(0)
        preds, loss = model(x, pred_steps=10, mode='adv_train')
        print(loss, len(preds), preds[0].size())
        optim.zero_grad()
        loss.backward()
        optim.step()