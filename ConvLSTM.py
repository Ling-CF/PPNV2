import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim + self.hidden_dim, 4 * self.hidden_dim, kernel_size=kernel_size, padding=padding, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4*self.hidden_dim, 4*self.hidden_dim, kernel_size=kernel_size, padding=padding, groups=4, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4 * self.hidden_dim, 4 * self.hidden_dim, kernel_size=kernel_size, padding=padding, groups=4, bias=True),
        )



        self.out_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.input_dim, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.relu = nn.LeakyReLU(inplace=True)


    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_dim, dim=1)

        i, f, o, g = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o), torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return self.out_conv(h_next), (h_next, c_next)



