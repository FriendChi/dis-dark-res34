import torch
from model.backbone import resnet,resnet1
import numpy as np
from utils.common import initialize_weights
from model.seg_model import SegHead
from model.layer import CoordConv

class parsingNet1(torch.nn.Module):
    def __init__(self, pretrained=True, backbone='50', num_grid_row = None, num_cls_row = None, num_grid_col = None, num_cls_col = None, num_lane_on_row = None, num_lane_on_col = None, use_aux=False,input_height = None, input_width = None, fc_norm = False):
        super(parsingNet1, self).__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col
        self.use_aux = use_aux
        #行方向网格数*行方向的车道线的类别数*行方向的车道线数数量
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        #2*行方向的车道线的类别数*行方向的车道线数数量
        #2是对存在性二元进行表达
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        mlp_mid_dim = 2048
        self.input_dim = input_height // 32 * input_width // 32 * 8

        self.model = resnet(backbone, pretrained=pretrained)

        # for avg pool experiment
        # self.pool = torch.nn.AdaptiveAvgPool2d(1)
        # self.pool = torch.nn.AdaptiveMaxPool2d(1)

        # self.register_buffer('coord', torch.stack([torch.linspace(0.5,9.5,10).view(-1,1).repeat(1,50), torch.linspace(0.5,49.5,50).repeat(10,1)]).view(1,2,10,50))

        #构建全连接层和线性层
        self.cls = torch.nn.Sequential(
            #fc_norm判断是否使用归一化层
            #Identity不对输入做任何改变
            #LayerNorm归一化操作，均值为0，方差为1
            torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),
            #线性变换，用于构建全连接层
            torch.nn.Linear(self.input_dim, mlp_mid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_mid_dim, self.total_dim),
        )
        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18', '34fca'] else torch.nn.Conv2d(2048,8,1)
        if self.use_aux:
            self.seg_head = SegHead(backbone, num_lane_on_row + num_lane_on_col)
        initialize_weights(self.cls)
    def forward(self, x):
        #骨干网络
        x2,x3,fea = self.model(x)
        #坐标卷积
        #是否使用坐标卷积
        '''
        print('x2 shape:',x2.shape)
        print('x3 shape:',x3.shape)
        print('fea shape:',fea.shape)
        '''
        if self.use_aux:
            seg_out = self.seg_head(x2, x3,fea)
        
        #print('fea shape:',fea.shape)
        #降维，将通道数降到8
        fea = self.pool(fea)
        # print(self.coord.shape)
        # fea = torch.cat([fea, self.coord.repeat(fea.shape[0],1,1,1)], dim = 1)
        #改变张量的形状，传入参数新shape
        fea = fea.view(-1, self.input_dim)
        out = self.cls(fea)
        #行方向上的位置预测、列方向上的位置预测、行方向上的存在性预测和列方向上的存在性预测
        #batchsize,行方向网格数100,行方向的车道线的类别数(锚点)56,行方向的车道线数数量4
        #loc_row[i,j,k,n]=v代表第i个样本中第j行第n条车道线的类别为第k个锚点的概率为v
        #exist_row[i,1,k,n]=v代表第i个样本中第n条车道线存在且类别为第k个锚点的概率为v
        pred_dict = {'loc_row': out[:,:self.dim1].view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row),
                     # batchsize,列方向网格数100,列方向的车道线的类别数(锚点)41,列方向的车道线数数量4
                'loc_col': out[:,self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
                #batchsize,2,56,4
                'exist_row': out[:,self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
                #batchsize,2,41,4
                'exist_col': out[:,-self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)}
        if self.use_aux:
            pred_dict['seg_out'] = seg_out
        
        return pred_dict

    def forward_tta(self, x):
        x2,x3,fea = self.model(x)

        pooled_fea = self.pool(fea)
        n,c,h,w = pooled_fea.shape
        #这样做的目的是为了增加模型对输入图像的平移不变性
        # 即当输入图像发生水平或垂直方向上的微小偏移时，模型的输出不会发生显著变化。
        #zeros_like创建一个形状、类型、内存布局和设备相同的零张量
        left_pooled_fea = torch.zeros_like(pooled_fea)
        right_pooled_fea = torch.zeros_like(pooled_fea)
        up_pooled_fea = torch.zeros_like(pooled_fea)
        down_pooled_fea = torch.zeros_like(pooled_fea)
        #pooled_fea的特征图整体左移，导致最右列空
        left_pooled_fea[:,:,:,:w-1] = pooled_fea[:,:,:,1:]
        #原最右列取平均赋值现在的空最右列
        left_pooled_fea[:,:,:,-1] = pooled_fea.mean(-1)
        
        right_pooled_fea[:,:,:,1:] = pooled_fea[:,:,:,:w-1]
        right_pooled_fea[:,:,:,0] = pooled_fea.mean(-1)

        up_pooled_fea[:,:,:h-1,:] = pooled_fea[:,:,1:,:]
        up_pooled_fea[:,:,-1,:] = pooled_fea.mean(-2)

        down_pooled_fea[:,:,1:,:] = pooled_fea[:,:,:h-1,:]
        down_pooled_fea[:,:,0,:] = pooled_fea.mean(-2)
        # 10 x 25  dim维度会发生变化
        fea = torch.cat([pooled_fea, left_pooled_fea, right_pooled_fea, up_pooled_fea, down_pooled_fea], dim = 0)
        fea = fea.view(-1, self.input_dim)

        out = self.cls(fea)

        return {'loc_row': out[:,:self.dim1].view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row), 
                'loc_col': out[:,self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
                'exist_row': out[:,self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
                'exist_col': out[:,-self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)}





class parsingNet(torch.nn.Module):
    def __init__(self, pretrained=True, backbone='50', num_grid_row = None, num_cls_row = None, num_grid_col = None, num_cls_col = None, num_lane_on_row = None, num_lane_on_col = None, use_aux=False,input_height = None, input_width = None, fc_norm = False):
        super(parsingNet, self).__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col
        self.use_aux = use_aux
        #行方向网格数*行方向的车道线的类别数*行方向的车道线数数量
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        #2*行方向的车道线的类别数*行方向的车道线数数量
        #2是对存在性二元进行表达
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        mlp_mid_dim = 2048
        self.input_dim = input_height // 32 * input_width // 32 * 8

        self.model = resnet1(backbone, pretrained=pretrained)

        # for avg pool experiment
        # self.pool = torch.nn.AdaptiveAvgPool2d(1)
        # self.pool = torch.nn.AdaptiveMaxPool2d(1)

        # self.register_buffer('coord', torch.stack([torch.linspace(0.5,9.5,10).view(-1,1).repeat(1,50), torch.linspace(0.5,49.5,50).repeat(10,1)]).view(1,2,10,50))

        #构建全连接层和线性层
        self.cls = torch.nn.Sequential(
            #fc_norm判断是否使用归一化层
            #Identity不对输入做任何改变
            #LayerNorm归一化操作，均值为0，方差为1
            torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),
            #线性变换，用于构建全连接层
            torch.nn.Linear(self.input_dim, mlp_mid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_mid_dim, self.total_dim),
        )
        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18', '34fca'] else torch.nn.Conv2d(2048,8,1)
        if self.use_aux:
            self.seg_head = SegHead(backbone, num_lane_on_row + num_lane_on_col)
        initialize_weights(self.cls)
    def forward(self, x):
        #骨干网络
        x2,x3,fea = self.model(x)
        #坐标卷积
        #是否使用坐标卷积
        '''
        print('x2 shape:',x2.shape)
        print('x3 shape:',x3.shape)
        print('fea shape:',fea.shape)
        '''
        if self.use_aux:
            seg_out = self.seg_head(x2, x3,fea)
        
        #print('fea shape:',fea.shape)
        #降维，将通道数降到8
        fea = self.pool(fea)
        # print(self.coord.shape)
        # fea = torch.cat([fea, self.coord.repeat(fea.shape[0],1,1,1)], dim = 1)
        #改变张量的形状，传入参数新shape
        fea = fea.view(-1, self.input_dim)
        out = self.cls(fea)
        #行方向上的位置预测、列方向上的位置预测、行方向上的存在性预测和列方向上的存在性预测
        #batchsize,行方向网格数100,行方向的车道线的类别数(锚点)56,行方向的车道线数数量4
        #loc_row[i,j,k,n]=v代表第i个样本中第j行第n条车道线的类别为第k个锚点的概率为v
        #exist_row[i,1,k,n]=v代表第i个样本中第n条车道线存在且类别为第k个锚点的概率为v
        pred_dict = {'loc_row': out[:,:self.dim1].view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row),
                     # batchsize,列方向网格数100,列方向的车道线的类别数(锚点)41,列方向的车道线数数量4
                'loc_col': out[:,self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
                #batchsize,2,56,4
                'exist_row': out[:,self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
                #batchsize,2,41,4
                'exist_col': out[:,-self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)}
        if self.use_aux:
            pred_dict['seg_out'] = seg_out
        
        return pred_dict

    def forward_tta(self, x):
        x2,x3,fea = self.model(x)

        pooled_fea = self.pool(fea)
        n,c,h,w = pooled_fea.shape
        #这样做的目的是为了增加模型对输入图像的平移不变性
        # 即当输入图像发生水平或垂直方向上的微小偏移时，模型的输出不会发生显著变化。
        #zeros_like创建一个形状、类型、内存布局和设备相同的零张量
        left_pooled_fea = torch.zeros_like(pooled_fea)
        right_pooled_fea = torch.zeros_like(pooled_fea)
        up_pooled_fea = torch.zeros_like(pooled_fea)
        down_pooled_fea = torch.zeros_like(pooled_fea)
        #pooled_fea的特征图整体左移，导致最右列空
        left_pooled_fea[:,:,:,:w-1] = pooled_fea[:,:,:,1:]
        #原最右列取平均赋值现在的空最右列
        left_pooled_fea[:,:,:,-1] = pooled_fea.mean(-1)
        
        right_pooled_fea[:,:,:,1:] = pooled_fea[:,:,:,:w-1]
        right_pooled_fea[:,:,:,0] = pooled_fea.mean(-1)

        up_pooled_fea[:,:,:h-1,:] = pooled_fea[:,:,1:,:]
        up_pooled_fea[:,:,-1,:] = pooled_fea.mean(-2)

        down_pooled_fea[:,:,1:,:] = pooled_fea[:,:,:h-1,:]
        down_pooled_fea[:,:,0,:] = pooled_fea.mean(-2)
        # 10 x 25  dim维度会发生变化
        fea = torch.cat([pooled_fea, left_pooled_fea, right_pooled_fea, up_pooled_fea, down_pooled_fea], dim = 0)
        fea = fea.view(-1, self.input_dim)

        out = self.cls(fea)

        return {'loc_row': out[:,:self.dim1].view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row), 
                'loc_col': out[:,self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
                'exist_row': out[:,self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
                'exist_col': out[:,-self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)}

def get_model(cfg):
    return parsingNet(pretrained = True, backbone=cfg.backbone, num_grid_row = cfg.num_cell_row, num_cls_row = cfg.num_row, num_grid_col = cfg.num_cell_col, num_cls_col = cfg.num_col, num_lane_on_row = cfg.num_lanes, num_lane_on_col = cfg.num_lanes, use_aux = cfg.use_aux, input_height = cfg.train_height, input_width = cfg.train_width, fc_norm = cfg.fc_norm).cuda()