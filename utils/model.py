import torch
import torch.nn as nn

CAPTCHA_LEN = 5 # len of captcha symbols (without .png)

class OCRmodel(nn.Module):

    def __init__(self, 
                 in_channels, 
                 cnn_out_dim,
                 seq_dim,
                 rnn_hidden_dim,
                 pad=1, 
                 strd=1, 
                 num_classes=None):
        super(OCRmodel, self).__init__()

        self.cnn = nn.Sequential(
              nn.Conv2d(in_channels=1, out_channels=64, 
                        kernel_size=3, stride=strd, padding=pad),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.BatchNorm2d(64),
              nn.ReLU(),
              nn.Dropout2d(0.1),
              nn.Conv2d(in_channels=64, out_channels=128, 
                        kernel_size=3, stride=strd, padding=pad),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.BatchNorm2d(128),
              nn.ReLU(),
              nn.Dropout2d(0.1),
              nn.Conv2d(in_channels=128, out_channels=256, 
                        kernel_size=3, stride=strd, padding=pad),
              nn.MaxPool2d(kernel_size=(1, 2), stride=2),
              nn.BatchNorm2d(256),
              nn.ReLU(),
              nn.Conv2d(in_channels=256, out_channels=512, 
                        kernel_size=3, stride=strd, padding=pad),
              nn.BatchNorm2d(512),
              nn.ReLU(),
              nn.Dropout2d(0.1),
              nn.MaxPool2d(kernel_size=(1, 2), stride=2),
              nn.Conv2d(in_channels=512, out_channels=512, 
                        kernel_size=2, stride=strd, padding=0),
              nn.BatchNorm2d(512),
              nn.ReLU(),
              nn.Dropout2d(0.1),
        )
        self.map_to_sec = nn.Linear(cnn_out_dim, seq_dim)
        self.rnn = nn.LSTM(input_size=seq_dim, 
                                      hidden_size=rnn_hidden_dim, 
                                      num_layers=3, 
                                      bidirectional=True)
        self.transcription = nn.Linear(2 * rnn_hidden_dim, num_classes)
        
    def forward(self, x):
        out = self.cnn(x)
        N, C, h, w = out.size()
        out = out.view(N, -1, w)
        out = out.permute(0, 2, 1)
        out = self.map_to_sec(out)
        out = out.permute(1, 0, 2)
        out, _ = self.rnn(out)
        out = self.transcription(out)
    
        input_lengths = torch.full(size=(out.size(1),), fill_value=out.size(0), dtype=torch.int32)
        target_lengths = torch.full(size=(out.size(1),), fill_value=CAPTCHA_LEN, dtype=torch.int32)
        return out, input_lengths, target_lengths
        