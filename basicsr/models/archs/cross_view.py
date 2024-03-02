import math
from tokenize import group
import torch
import torch.nn as nn
import torch.nn.functional as F
#leftview PSNR: 35.3590 SSIM: 0.9487 MAE: 0.0151
#rightview PSNR: 35.9376 SSIM: 0.9451 MAE: 0.0132
#On Flickr1024:
#XZXX
#===================
#New dataset:ALLIN
class attention2d(nn.Module):
    def __init__(self, in_planes, K,):
        super(attention2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, K, 1,)
        self.fc2 = nn.Conv2d(K, K, 1,)
    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x, 1)
class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, K=3, ):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, K, )

        self.weight = nn.Parameter(torch.Tensor(K, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None

    def forward(self, x,y): 
        softmax_attention = self.attention(y)
        batch_size, in_planes, height, width = x.size()
        x = x.reshape(1, -1, height, width)  
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size,
                                                                    self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.l_func = Dynamic_conv2d(in_planes=c,out_planes=c,kernel_size=1,stride=1,padding=0)
        self.r_func = Dynamic_conv2d(in_planes=c,out_planes=c,kernel_size=1,stride=1,padding=0)

#B,C,H,W
    def forward(self, x_l, x_r):
        shortcut_l = torch.tensor(4)
        shortcut_r = torch.tensor(4)
        shortcut_lr = torch.tensor(4)
        shortcut_rl = torch.tensor(4)
        weach = 10
        tempe = x_r.shape[3]
        # print("tempe.",tempe)
        # rangenum = math.ceil(tempe / weach)
        rangenum = tempe//weach
        for i in range(0,rangenum):
            index = torch.tensor([ i * weach + j for j in range(0,weach)]).cuda()
            x_l1 = torch.index_select(x_l,3,index=index).cuda()
            x_r1 = torch.index_select(x_r,3,index=index).cuda()
            Q_l = self.l_proj1(self.norm_l(x_l1)).permute(0, 2, 3, 1)  # B, H, W, c
            Q_r_T = self.r_proj1(self.norm_r(x_r1)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)
            V_l = self.l_proj2(x_l1).permute(0, 2, 3, 1)  # B, H, W, c
            V_r = self.r_proj2(x_r1).permute(0, 2, 3, 1)  # B, H, W, c
            # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
            attention = torch.matmul(Q_l, Q_r_T) * self.scale
            F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
            attention = attention.permute(0,1,3,2)
            attentiontmp = torch.softmax(attention,dim=-1)
            F_l2r = torch.matmul(attentiontmp, V_l)  # B, H, W, c
            # scale
            F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
            F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
            if i == 0:
                shortcut_l = x_l1
                shortcut_r = x_r1
                shortcut_lr = F_l2r
                shortcut_rl = F_r2l
            else:
                shortcut_l = torch.cat((shortcut_l,x_l1),dim=3)
                shortcut_r = torch.cat((shortcut_r,x_r1),dim=3)
                shortcut_lr = torch.cat((shortcut_lr,F_l2r),dim=3)
                shortcut_rl = torch.cat((shortcut_rl,F_r2l),dim=3)
       # print("shortcut_rl.shape",shortcut_rl.shape)
        shortcut_rl = self.l_func(shortcut_rl,shortcut_l)
        A = shortcut_l + shortcut_rl
        shortcut_lr = self.r_func(shortcut_lr,shortcut_r)
        B = shortcut_r + shortcut_lr
        return A, B


class PUNet(nn.Module):
    def __init__(self, depth=[2, 2, 2, 2, 2]):
        super(PUNet, self).__init__()

        base_channel = 32

        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[LKB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel * 2, base_channel * 2, 3, 1),
            nn.Sequential(*[LKB(base_channel * 2) for _ in range(depth[1])]),
            Down_scale(base_channel * 2),
            BasicConv(base_channel * 4, base_channel * 4, 3, 1),
            nn.Sequential(*[LKB(base_channel * 4) for _ in range(depth[2])]),
            Down_scale(base_channel * 4),
        ])

        # Middle
        self.middle = nn.Sequential(*[LKB(base_channel * 8) for _ in range(depth[3])])

        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale(base_channel * 8),
            BasicConv(base_channel * 8, base_channel * 4, 3, 1),
            nn.Sequential(*[LKB(base_channel * 4) for _ in range(depth[2])]),
            Up_scale(base_channel * 4),
            BasicConv(base_channel * 4, base_channel * 2, 3, 1),
            nn.Sequential(*[LKB(base_channel * 2) for _ in range(depth[1])]),
            Up_scale(base_channel * 2),
            BasicConv(base_channel * 2, base_channel, 3, 1),
            nn.Sequential(*[LKB(base_channel) for _ in range(depth[0])]),
        ])
        # conv
        self.conv_first = BasicConv(3, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 3, 3, 1, 1)
        self.refine = nn.Sequential(*[LKB(base_channel) for i in range(depth[4])])
        self.fusion = nn.ModuleList([SCAM(base_channel),
                                     SCAM(base_channel*2),
                                     SCAM(base_channel*4),
                                     SCAM(base_channel*8)])

    def skip_con(self, shortcuts_l, shortcuts_r):
        refine_l, refine_r = [], []
        for i in range(len(shortcuts_l)):
            shortcut_l, shortcut_r = shortcuts_l[i], shortcuts_r[i]
            # stereo fusion

            refine_l.append(shortcut_l)
            refine_r.append(shortcut_r)
        return refine_l, refine_r

    def encoder(self, x, y):
        shortcuts_x = []
        shortcuts_y = []
        for i in range(len(self.Encoder)):
            index = (i+2)//3 - 1
            if (i + 2) % 3 == 0:
                x,y = self.fusion[index](x,y)
                x = self.Encoder[i](x)
                y = self.Encoder[i](y)
                shortcuts_x.append(x)
                shortcuts_y.append(y)
            else:
                x = self.Encoder[i](x)
                y = self.Encoder[i](y)
        return shortcuts_x, x, shortcuts_y, y

    def decoder(self, x, shortcuts_x, y, shortcuts_y):
        index_x, index_y = len(shortcuts_x), len(shortcuts_y)
        for i in range(len(self.Decoder)):
            index = (i+1)//3
            if (i + 2) % 3 == 0:
                index_x = index_x - 1
                index_y = index_y - 1
                x, y = torch.cat([x, shortcuts_x[index_x]], 1), torch.cat([y, shortcuts_y[index_y]], 1)
            if (i + 1)% 3 == 0:
                x , y = self.fusion[3-index](x,y)
                x, y = self.Decoder[i](x), self.Decoder[i](y)
            else:
                x, y = self.Decoder[i](x), self.Decoder[i](y)
        return x, y

    def to_one(self, x):
        return (torch.tanh(x) + 1) / 2

    def forward(self, x):
        x_l, x_r = x[:, :3, :, :], x[:, 3:, :, :]
        x_l, x_r = self.conv_first(x_l), self.conv_first(x_r)
        shortcuts_l, x_l, shortcuts_r, x_r = self.encoder(x_l, x_r)
        x_l, x_r = self.middle(x_l), self.middle(x_r)
        shortcuts_l, shortcuts_r = self.skip_con(shortcuts_l, shortcuts_r)
        x_l, x_r = self.decoder(x_l, shortcuts_l, x_r, shortcuts_r)
        x_l, x_r = self.refine(x_l), self.refine(x_r)
        x_l, x_r = self.conv_last(x_l), self.conv_last(x_r)
        x_l, x_r = self.to_one(x_l), self.to_one(x_r)
        return x_l, x_r
class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x

def M_Relax(M, num_pixels):
    _, u, v = M.shape
    M_list = []
    M_list.append(M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
        pad_M = pad(M[:, :-1-i, :])
        M_list.append(pad_M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
        pad_M = pad(M[:, i+1:, :])
        M_list.append(pad_M.unsqueeze(1))
    M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
    return M_relaxed

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, activation=True,
                 transpose=False, groups=True):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            # layers.append(nn.LayerNorm(out_channel))
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class GroupConv(nn.Module):
    def __init__(self, dim, kernel_size, stride=1):
        super(GroupConv, self).__init__()
        padding = kernel_size // 2
        self.main = nn.Conv2d(dim, dim, kernel_size, padding=padding, stride=stride, bias=True, groups=dim)

    def forward(self, x):
        return self.main(x)


class MLP(nn.Module):
    def __init__(self, dim_in, din_out):
        super(MLP, self).__init__()
        self.main = nn.Conv2d(dim_in, din_out, 1, 1)

    def forward(self, x):
        return self.main(x)


class LKB(nn.Module):
    def __init__(self, dim):
        super(LKB, self).__init__()
        self.layer_I = nn.Sequential(*[
            LayerNorm2d(dim),
            MLP(dim, dim * 2),

        ])
        self.layer_1 = nn.Sequential(*[
            GroupConv(dim * 2, 1),
            MLP(dim * 2, dim//2)
        ])
        self.layer_2 = nn.Sequential(*[
            GroupConv(dim * 2, 3),
            MLP(dim * 2, dim // 2)
        ])
        self.layer_3 = nn.Sequential(*[
            GroupConv(dim * 2, 5),
            MLP(dim * 2, dim // 2)

        ])
        self.layer_4 = nn.Sequential(*[
            GroupConv(dim * 2, 7),
            MLP(dim * 2, dim // 2)

        ])
        self.layer_II = nn.Sequential(*[
            MLP(dim * 2, dim)
        ])
        self.layer_III = nn.Sequential(*[
            LayerNorm2d(dim),
            MLP(dim, dim * 2),
            nn.GELU(),
            MLP(dim * 2, dim)
        ])

    def forward(self, x):
        x0 = self.layer_I(x)
        x1 = self.layer_1(x0)
        x2 = self.layer_2(x0)
        x3 = self.layer_3(x0)
        x4 = self.layer_4(x0)
        x_mid = torch.cat((x1,x2,x3,x4),dim=1)
        x_mid = self.layer_II(x_mid)
        x_mid = x_mid + x
        x = self.layer_III(x_mid) + x_mid
        return x


class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)

    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x


class RDB(nn.Module):
    def __init__(self, dim, gd=32):
        super(RDB, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = BasicConv(dim, gd, 3, 1)
        self.conv2 = BasicConv(dim + gd, gd, 3, 1)
        self.conv3 = BasicConv(dim + gd * 2, gd, 3, 1)
        self.conv4 = BasicConv(dim + gd * 3, gd, 3, 1)
        self.conv5 = BasicConv(dim + gd * 4, dim, 3, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, dim, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = RDB(dim, gc)
        self.RDB2 = RDB(dim, gc)
        self.RDB3 = RDB(dim, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class CSI(nn.Module):
    def __init__(self, dim):
        # cross-scale interaraction
        super(CSI, self).__init__()
        self.low_up1 = nn.Sequential(*[Up_scale(dim * 4), RRDB(dim * 2)])
        self.low_up2 = nn.Sequential(*[Up_scale(dim * 2), RRDB(dim)])

        self.mid_up = nn.Sequential(*[Up_scale(dim * 2), RRDB(dim)])
        self.mid_down = nn.Sequential(*[Down_scale(dim * 2), RRDB(dim * 4)])

        self.high_down1 = nn.Sequential(*[Down_scale(dim), RRDB(dim * 2)])
        self.high_down2 = nn.Sequential(*[Down_scale(dim * 2), RRDB(dim * 4)])

        self.conv_l = BasicConv(dim * 12, dim * 4, 1, 1)
        self.conv_m = BasicConv(dim * 6, dim * 2, 1, 1)
        self.conv_h = BasicConv(dim * 3, dim, 1, 1)

    def forward(self, shortcuts):
        high, mid, low = shortcuts[0], shortcuts[1], shortcuts[2]

        l2m = self.low_up1(low)
        l2h = self.low_up2(l2m)
        m2h = self.mid_up(mid)
        m2l = self.mid_down(mid)
        h2m = self.high_down1(high)
        h2l = self.high_down2(h2m)

        low = self.conv_l(torch.cat([low, m2l, h2l], 1))
        mid = self.conv_m(torch.cat([l2m, mid, h2m], 1))
        high = self.conv_h(torch.cat([l2h, m2h, high], 1))

        return [high, mid, low]


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel * 2, 3, 2)

    def forward(self, x):
        return self.main(x)


class Up_scale(nn.Module):
    def __init__(self, in_channel):
        super(Up_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel // 2, kernel_size=4, activation=True, stride=2, transpose=True)

    def forward(self, x):
        return self.main(x)

