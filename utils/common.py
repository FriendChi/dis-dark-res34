import os, argparse
from data.dali_data import TrainCollect
from utils.dist_utils import get_rank, get_world_size, is_main_process, dist_print, DistSummaryWriter
from utils.config import Config
import torch
import time
import json

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    #创建一个解析器对象
    parser = argparse.ArgumentParser()
    #向解析器中添加一个参数或选项，它可以指定参数或选项的名称、类型、默认值、帮助信息等
    parser.add_argument('config', help = 'path to config file')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--dataset', default = None, type = str)
    parser.add_argument('--data_root', default = None, type = str)
    parser.add_argument('--epoch', default = None, type = int)
    parser.add_argument('--batch_size', default = None, type = int)
    parser.add_argument('--optimizer', default = None, type = str)
    parser.add_argument('--learning_rate', default = None, type = float)
    parser.add_argument('--weight_decay', default = None, type = float)
    parser.add_argument('--momentum', default = None, type = float)
    parser.add_argument('--scheduler', default = None, type = str)
    parser.add_argument('--steps', default = None, type = int, nargs='+')
    parser.add_argument('--gamma', default = None, type = float)
    parser.add_argument('--warmup', default = None, type = str)
    parser.add_argument('--warmup_iters', default = None, type = int)
    parser.add_argument('--backbone', default = None, type = str)
    parser.add_argument('--griding_num', default = None, type = int)
    parser.add_argument('--use_aux', default = None, type = str2bool)
    parser.add_argument('--sim_loss_w', default = None, type = float)
    parser.add_argument('--shp_loss_w', default = None, type = float)
    parser.add_argument('--note', default = None, type = str)
    parser.add_argument('--log_path', default = None, type = str)
    parser.add_argument('--finetune', default = None, type = str)
    parser.add_argument('--resume', default = None, type = str)
    parser.add_argument('--test_model', default = None, type = str)
    parser.add_argument('--test_work_dir', default = None, type = str)
    parser.add_argument('--num_lanes', default = None, type = int)
    parser.add_argument('--auto_backup', action='store_false', help='automatically backup current code in the log path')
    parser.add_argument('--var_loss_power', default = None, type = float)
    parser.add_argument('--num_row', default = None, type = int)
    parser.add_argument('--num_col', default = None, type = int)
    parser.add_argument('--train_width', default = None, type = int)
    parser.add_argument('--train_height', default = None, type = int)
    parser.add_argument('--num_cell_row', default = None, type = int)
    parser.add_argument('--num_cell_col', default = None, type = int)
    parser.add_argument('--mean_loss_w', default = None, type = float)
    parser.add_argument('--fc_norm', default = None, type = str2bool)
    parser.add_argument('--soft_loss', default = None, type = str2bool)
    parser.add_argument('--cls_loss_col_w', default = None, type = float)
    parser.add_argument('--cls_ext_col_w', default = None, type = float)
    parser.add_argument('--mean_loss_col_w', default = None, type = float)
    parser.add_argument('--eval_mode', default = None, type = str)
    parser.add_argument('--eval_during_training', default = None, type = str2bool)
    parser.add_argument('--split_channel', default = None, type = str2bool)
    parser.add_argument('--match_method', default = None, type = str, choices = ['fixed', 'hungarian'])
    parser.add_argument('--selected_lane', default = None, type = int, nargs='+')
    parser.add_argument('--cumsum', default = None, type = str2bool)
    parser.add_argument('--masked', default = None, type = str2bool)
    
    
    return parser

import numpy as np
def merge_config():
    args = get_args().parse_args()
    #从配置文件中读取配置信息，并得到一个Config对象cfg
    cfg = Config.fromfile(args.config)
    #定义了一个列表items，它包含了一些可能需要修改的配置项的名称
    items = ['dataset','data_root','epoch','batch_size','optimizer','learning_rate',
    'weight_decay','momentum','scheduler','steps','gamma','warmup','warmup_iters',
    'use_aux','griding_num','backbone','sim_loss_w','shp_loss_w','note','log_path',
    'finetune','resume', 'test_model','test_work_dir', 'num_lanes', 'var_loss_power', 'num_row', 'num_col', 'train_width', 'train_height',
    'num_cell_row', 'num_cell_col', 'mean_loss_w','fc_norm','soft_loss','cls_loss_col_w', 'cls_ext_col_w', 'mean_loss_col_w', 'eval_mode', 'eval_during_training', 'split_channel', 'match_method', 'selected_lane', 'cumsum', 'masked']
    #遍历items中的每个配置项
    # 如果args中有对应的属性，并且不是None，就用args中的属性值覆盖cfg中的属性值，并打印出合并的信息
    for item in items:
        if getattr(args, item) is not None:
            dist_print('merge ', item, ' config')
            setattr(cfg, item, getattr(args, item))

    if cfg.dataset == 'CULane':
        cfg.row_anchor = np.linspace(0.42,1, cfg.num_row)
        cfg.col_anchor = np.linspace(0,1, cfg.num_col)
    elif cfg.dataset == 'Tusimple':
        cfg.row_anchor = np.linspace(160,710, cfg.num_row)/720
        cfg.col_anchor = np.linspace(0,1, cfg.num_col)
    elif cfg.dataset == 'CurveLanes':
        cfg.row_anchor = np.linspace(0.4, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    
    return args, cfg


def save_model(net, optimizer, epoch,save_path, distributed):
    if is_main_process():
        model_state_dict = net.state_dict()
        #调优化器和模型的参数
        state = {'model': model_state_dict, 'optimizer': optimizer.state_dict()}
        # state = {'model': model_state_dict}

        assert os.path.exists(save_path)

        # 获取指定目录下的所有文件
        files = os.listdir(save_path)
        for file in files:
            # 判断文件是否为.pth文件，并且不是当前要保存的检查点文件
            if file.endswith('.pth'):
                file_path = os.path.join(save_path, file)
                # 删除文件
                os.remove(file_path)

        model_path = os.path.join(save_path, f'model_{epoch:03d}.pth')
        #model_path = os.path.join(save_path, 'model_best.pth')
        torch.save(state, model_path)

import pathspec

def cp_projects(auto_backup, to_path):
    if is_main_process() and auto_backup:
        with open('./.gitignore','r') as fp:
            ign = fp.read()
        ign += '\n.git'
        spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ign.splitlines())
        all_files = {os.path.join(root,name) for root,dirs,files in os.walk('./') for name in files}
        matches = spec.match_files(all_files)
        matches = set(matches)
        to_cp_files = all_files - matches
        dist_print('Copying projects to '+ to_path + ' for backup')
        t0 = time.time()
        warning_flag = True
        for f in to_cp_files:
            dirs = os.path.join(to_path,'code',os.path.split(f[2:])[0])
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            os.system('cp %s %s'%(f,os.path.join(to_path,'code',f[2:])))
            elapsed_time = time.time() - t0
            if elapsed_time > 5 and warning_flag:
                dist_print('If the program is stuck, it might be copying large files in this directory. please don\'t set --auto_backup. Or please make you working directory clean, i.e, don\'t place large files like dataset, log results under this directory.')
                warning_flag = False




import datetime, os
def get_work_dir(cfg):
    #获取当前时间并将其格式化为字符串，格式为年月日_时分秒
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    #根据配置信息cfg中的学习率和批次大小，生成一个超参数字符串hyper_param_str。
    # %1.0e表示1位整数、0位小数的浮点数格式，%d表示10进制整数
    hyper_param_str = '_lr_%1.0e_b_%d' % (cfg.learning_rate, cfg.batch_size)
    #将三个字符串拼接成工作目录路径work_dir。
    # 其中cfg.log_path表示日志文件的保存路径，cfg.note表示用户自定义的备注信息
    work_dir = os.path.join(cfg.log_path, now + hyper_param_str + cfg.note)
    return work_dir

def get_logger(work_dir, cfg):
    logger = DistSummaryWriter(work_dir)
    config_txt = os.path.join(work_dir, 'cfg.txt')
    if is_main_process():
        with open(config_txt, 'w') as fp:
            fp.write(str(cfg))

    return logger

def initialize_weights(*models):
    for model in models:
        real_init_weights(model)
def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)
            
import importlib
def get_model(cfg):
    #根据cfg中的dataset属性，拼接出一个模块名，比如'model.model_tusimple'
    #使用importlib.import_module()函数，动态地导入这个模块，并返回一个模块对象。
    #调用这个模块对象的get_model(cfg)函数，传入cfg作为参数，并返回一个模型对象。
    return importlib.import_module('model.model_'+cfg.dataset.lower()).get_model(cfg)

def get_train_loader(cfg):
    if cfg.dataset == 'CULane':
        train_loader = TrainCollect(cfg.batch_size, 4, cfg.data_root, os.path.join(cfg.data_root, 'list/train_gt.txt'), get_rank(), get_world_size(), 
                                cfg.row_anchor, cfg.col_anchor, cfg.train_width, cfg.train_height, cfg.num_cell_row, cfg.num_cell_col, cfg.dataset, cfg.crop_ratio)
    elif cfg.dataset == 'Tusimple':
#get_world_size()用于获取分布式计算集群的大小。如果分布式计算环境不可用或者未初始化则返回1，表示没有分布式环境
#get_rank()若不用分布式则返回0
        train_loader = TrainCollect(cfg.batch_size, 4, cfg.data_root, os.path.join(cfg.data_root, 'train_gt.txt'), get_rank(), get_world_size(), 
                                cfg.row_anchor, cfg.col_anchor, cfg.train_width, cfg.train_height, cfg.num_cell_row, cfg.num_cell_col, cfg.dataset, cfg.crop_ratio)
    elif cfg.dataset == 'CurveLanes':
        train_loader = TrainCollect(cfg.batch_size, 4, cfg.data_root, os.path.join(cfg.data_root, 'train', 'train_gt.txt'), get_rank(), get_world_size(), 
                                cfg.row_anchor, cfg.col_anchor, cfg.train_width, cfg.train_height, cfg.num_cell_row, cfg.num_cell_col, cfg.dataset, cfg.crop_ratio)
    else:
        raise NotImplementedError
    return train_loader 

def inference(net, teacher_net,data_label, dataset):
    if dataset == 'CurveLanes':
        return inference_curvelanes(net, data_label)
    elif dataset in ['Tusimple', 'CULane']:
        return inference_culane_tusimple(net,teacher_net, data_label)
    else:
        raise NotImplementedError

n=0
def write_json(name,data):
    global n
    n+=1
    if n>14:
        print('结束写入')
        return 
    # 指定JSON文件路径
    file_path = str(n)+'data.json'

    # 读取JSON文件中的数据
    try:
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}

    # 新数据
    new_data = {
        'name': name,
        'data': data.tolist()
    }

    # 合并数据
    existing_data.update(new_data)

    # 将合并后的数据写入JSON文件
    with open(file_path, 'a') as file:
        json.dump(existing_data, file, indent=4)

    print(name,"新数据已成功添加到 JSON 文件中。")

def inference_culane_tusimple(net,teacher_net, data_label):
    #得到模型输出结果
    pred = net(data_label['images'])
    teacher_pred = teacher_net(data_label['images'])

    # #获取行方向的标签数据，判断它们是否不等于-1
    cls_out_ext_label = (data_label['labels_row'] != -1).long()
    #获取列方向的标签数据，判断它们是否不等于-1
    cls_out_col_ext_label = (data_label['labels_col'] != -1).long()
    
    # write_json("pred['loc_row']",pred['loc_row'])
    # write_json("data_label['labels_row']",data_label['labels_row'])
    # write_json("pred['loc_col']",pred['loc_col'])
    # write_json("data_label['labels_col']",data_label['labels_col'])
    # write_json("pred['exist_row']",pred['exist_row'])
    # write_json("cls_out_ext_label",cls_out_ext_label)
    # write_json("pred['exist_col']",pred['exist_col'])
    # write_json("cls_out_col_ext_label",cls_out_col_ext_label)
    # write_json("data_label['labels_row_float']",data_label['labels_row_float'])
    # write_json("data_label['labels_col_float']",data_label['labels_col_float'])
    # write_json("pred['loc_row']",teacher_pred['loc_row'])
    # write_json("pred['loc_col']",teacher_pred['loc_col'])
    # write_json("pred['exist_row']",teacher_pred['exist_row'])
    # write_json("pred['exist_col']",teacher_pred['exist_col'])
    

    #将模型输出结果和数据标签字典中的一些数据赋值给它的不同键
    #4个模型输出，6个标签，其中4个标签有对应输出
    res_dict = {'cls_out': pred['loc_row'], 'cls_label': data_label['labels_row'], 'cls_out_col':pred['loc_col'],'cls_label_col':data_label['labels_col'],
            'cls_out_ext':pred['exist_row'], 'cls_out_ext_label':cls_out_ext_label, 'cls_out_col_ext':pred['exist_col'],
                'cls_out_col_ext_label':cls_out_col_ext_label, 'labels_row_float':data_label['labels_row_float'], 'labels_col_float':data_label['labels_col_float'], 
                'cls_out_col_teacher': teacher_pred['loc_col'],'cls_out_teacher': teacher_pred['loc_row']}
    if 'seg_out' in pred.keys():
        res_dict['seg_out'] = pred['seg_out']
        res_dict['seg_label'] = data_label['seg_images']

    return res_dict
def inference_curvelanes(net, data_label):
    pred = net(data_label['images'])
    cls_out_ext_label = (data_label['labels_row'] != -1).long()
    cls_out_col_ext_label = (data_label['labels_col'] != -1).long()

    res_dict = {'cls_out': pred['loc_row'], 'cls_label': data_label['labels_row'], 'cls_out_col':pred['loc_col'],'cls_label_col':data_label['labels_col'],
                'cls_out_ext':pred['exist_row'], 'cls_out_ext_label':cls_out_ext_label, 'cls_out_col_ext':pred['exist_col'],
                'cls_out_col_ext_label':cls_out_col_ext_label, 'seg_label': data_label['seg_images'], 'seg_out_row': pred['lane_token_row'], 'seg_out_col': pred['lane_token_col'] }
    if 'seg_out' in pred.keys():
        res_dict['seg_out'] = pred['seg_out']
        res_dict['seg_label'] = data_label['segs']
    return res_dict






'''
res_dict 
{'cls_out': pred['loc_row'], 'cls_label': data_label['labels_row'], 'cls_out_col':pred['loc_col'],
 'cls_label_col':data_label['labels_col'],'cls_out_ext':pred['exist_row'], 'cls_out_ext_label':cls_out_ext_label,
 'cls_out_col_ext':pred['exist_col'],'cls_out_col_ext_label':cls_out_col_ext_label,
'labels_row_float':data_label['labels_row_float'], 'labels_col_float':data_label['labels_col_float'] 
    ,'seg_out': pred['seg_out'],'seg_label':data_label['seg_images']}
'''
def calc_loss(loss_dict, results, logger, global_step, epoch):
    loss = 0

    for i in range(len(loss_dict['name'])):
        #判断当前损失函数的权重系数是否为0，
        # 如果是，就跳过这个损失函数，继续下一个循环
        if loss_dict['weight'][i] == 0:
            continue
            
        data_src = loss_dict['data_src'][i]

        #训练得到的数据结果
        datas = [results[src] for src in data_src]
        #调用当前损失函数的实例化对象，传入提取的数据作为参数，计算当前损失值
        #data_src：('seg_out_row', 'seg_label'), ('seg_out_col', 'seg_label')
        #datas：[results[seg_out_row],results[seg_label]]
        loss_cur = loss_dict['op'][i](*datas)

        #判断全局步数是否能被20整除，如果是就用日志对象记录当前损失函数的名称，当前损失值和全局步数
        if global_step % 20 == 0:
            # print([src for src in data_src])
            # print([results[src].shape for src in data_src])
            # print(loss_dict['op'][i].__class__.__name__)
            # print(loss_cur)
            # print('---------------')
            logger.add_scalar('loss/'+loss_dict['name'][i], loss_cur, global_step)
        #将当前损失值乘以权重系数，并累加到总的损失值上，并返回
        loss += loss_cur * loss_dict['weight'][i]

    return loss