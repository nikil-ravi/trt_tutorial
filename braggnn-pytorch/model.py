import torch

def model_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class NLB(torch.nn.Module):
    def __init__(self, in_ch, relu_a=0.01):
        # TODO: bring this back self.inter_ch = torch.div(in_ch, 2, rounding_mode='floor').item()
        self.inter_ch = int(torch.floor(torch.div(in_ch, 2)).item())
        super().__init__()
        self.theta_layer = torch.nn.Conv2d(in_channels=in_ch, out_channels=self.inter_ch, \
                            kernel_size=1, padding=0)
        self.phi_layer   = torch.nn.Conv2d(in_channels=in_ch, out_channels=self.inter_ch, \
                            kernel_size=1, padding=0)
        self.g_layer     = torch.nn.Conv2d(in_channels=in_ch, out_channels=self.inter_ch, \
                            kernel_size=1, padding=0)
        self.atten_act   = torch.nn.Softmax(dim=-1)
        self.out_cnn     = torch.nn.Conv2d(in_channels=self.inter_ch, out_channels=in_ch, \
                            kernel_size=1, padding=0)
        
    def forward(self, x):
        mbsz, _, h, w = x.size()
        
        theta = self.theta_layer(x).view(mbsz, self.inter_ch, -1).permute(0, 2, 1)
        phi   = self.phi_layer(x).view(mbsz, self.inter_ch, -1)
        g     = self.g_layer(x).view(mbsz, self.inter_ch, -1).permute(0, 2, 1)
        
        theta_phi = self.atten_act(torch.matmul(theta, phi))
        
        theta_phi_g = torch.matmul(theta_phi, g).permute(0, 2, 1).view(mbsz, self.inter_ch, h, w)
        
        _out_tmp = self.out_cnn(theta_phi_g)
        _out_tmp = torch.add(_out_tmp, x)
   
        return _out_tmp


class BraggNN(torch.nn.Module):
    def __init__(self, imgsz, fcsz=(64, 32, 16, 8)):
        super().__init__()
        self.cnn_ops = []
        cnn_out_chs = (64, 32, 8)
        cnn_in_chs  = (1, ) + cnn_out_chs[:-1]
        fsz = imgsz
        for ic, oc, in zip(cnn_in_chs, cnn_out_chs):
            self.cnn_ops += [
                            torch.nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=3, \
                                stride=1, padding=0),
                            torch.nn.LeakyReLU(negative_slope=0.01),
            ]
            fsz -= 2
        self.nlb = NLB(in_ch=cnn_out_chs[0])
        self.dense_ops = []
        dense_in_chs  = (fsz * fsz * cnn_out_chs[-1], ) + fcsz[:-1]
        for ic, oc in zip(dense_in_chs, fcsz):
            self.dense_ops += [
                            torch.nn.Linear(ic, oc),
                            torch.nn.LeakyReLU(negative_slope=0.01),
            ]
        # output layer
        self.dense_ops += [torch.nn.Linear(fcsz[-1], 2), ]
        
        self.cnn_layers   = torch.nn.Sequential(*self.cnn_ops)
        self.dense_layers = torch.nn.Sequential(*self.dense_ops)
        
    def forward(self, x):
        _out = x
        for layer in self.cnn_layers[:1]:
            _out = layer(_out)

        _out = self.nlb(_out)

        for layer in self.cnn_layers[1:]:
            _out = layer(_out)
        
        _out = _out.flatten(start_dim=1)
        for layer in self.dense_layers:
            _out = layer(_out)
            
        return _out