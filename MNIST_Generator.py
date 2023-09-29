import torch
import torch.nn as nn

generator_out_linear = 100 #l rumore che andrà al generatore sarà 100+generator_out_linear, deve essere >=10

# Generator model
class MNIST_Generator(nn.Module):
    def __init__(self, z_dim = 100, label_dim = 10):
        super(MNIST_Generator, self).__init__()

        self.ylabel=nn.Sequential(
            nn.Linear(label_dim, generator_out_linear),
            nn.ReLU(True)
        )

        self.concat = nn.Sequential(
            #il rumore diventerà z_dim + il rumore della condizione
            nn.ConvTranspose2d(z_dim+generator_out_linear, 64*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False),    
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, y):
        
        y=y.reshape(-1,10)
        y = self.ylabel(y)
        y=y.reshape(-1,generator_out_linear,1,1)

        out = torch.cat([x, y] , dim=1)
        out=out.view(-1,100+generator_out_linear,1,1)

        out = self.concat(out)
        
        return out

