import torch, os, datetime


from utils.dist_utils import dist_print, dist_tqdm, synchronize
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics

from utils.common import calc_loss, get_model, get_train_loader, inference, merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

import time
from evaluation.eval_wrapper import eval_lane
'''
python train.py 配置文件的路径 --log_path 保存日志和模型的路径
'''
#用来在一个数据加载器上进行一轮训练，计算损失函数，更新优化器，记录日志和指标等
def train(net,teacher_net, data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, dataset):
    #将网络设置成训练模式
    net.train()
    teacher_net.eval()
    #创建进度条
    progress_bar = dist_tqdm(train_loader)
    #r循环遍历进度条对象中的每个元素，每个元素都是一个包含了数据和标签（data and label）的元组（tuple）
    for b_idx, data_label in enumerate(progress_bar):
        #b_idx是迭代进度条中的计数器
        global_step = epoch * len(data_loader) + b_idx
        #返回的是预测得到的数据
        #dataset是名称
        results = inference(net, teacher_net,data_label, dataset)
        #计算损失值
        loss = calc_loss(loss_dict, results, logger, global_step, epoch)
        #将优化器中的梯度缓存清零，防止累计梯度
        optimizer.zero_grad()
        #根据损失值计算网络梯度
        loss.backward()
        #更新网络参数
        optimizer.step()
        #根据当前训练步数（global_step）更新学习率（learning rate）
        scheduler.step(global_step)


        # if global_step % 20 == 0:
        #     reset_metrics(metric_dict)
        #     #update_metrics(metric_dict, results)
        #     for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
        #         logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
        #     logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        #     if hasattr(progress_bar,'set_postfix'):
        #         kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
        #         new_kwargs = {}
        #         for k,v in kwargs.items():
        #             if 'lane' in k:
        #                 continue
        #             new_kwargs[k] = v
        #         progress_bar.set_postfix(loss = '%.3f' % float(loss), 
        #                                 **new_kwargs)
        
#用来设置一些参数，初始化模型，数据加载器，优化器，学习率调度器，日志器等，并根据是否分布式训练进行相应的配置。
# 如果有指定 finetune 或 resume 的模型路径，就从中加载模型权重。然后对每个 epoch 进行训练和评估，并保存最佳的模型。
if __name__ == "__main__":
    #用来设置是否使用cudnn的自动优化功能，如果为True，就会根据网络的输入和输出选择最快的算法。
    torch.backends.cudnn.benchmark = True
    #合并命令行参数和配置文件的参数
    args, cfg = merge_config()
    #判断是否是第一个进程，如果是，就调用get_work_dir函数，根据cfg中的参数生成一个工作目录，并赋值给work_dir变量。
    if args.local_rank == 0:
        work_dir = get_work_dir(cfg)
    #是否使用分布式训练，默认为False
    distributed = False
    #如果环境变量中有WORLD_SIZE，并且它的值大于1，那么就说明有多个进程参与训练，需要进行分布式训练。
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        if args.local_rank == 0:
            with open('.work_dir_tmp_file.txt', 'w') as f:
                f.write(work_dir)
        else:
            while not os.path.exists('..txt'):
                time.sleep(0.1)
            with open('.work_dir_tmp_file.txt', 'r') as f:
                work_dir = f.read().strip()
    #同步所有进程
    synchronize()
    #更新cfg变量
    cfg.test_work_dir = work_dir
    cfg.distributed = distributed
    #用来在第一个进程中删除临时文件
    if args.local_rank == 0:
        os.system('rm .work_dir_tmp_file.txt')
    #用来打印当前时间和开始训练的信息，只有第一个进程会打印，其他进程不会打印。
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide', '34fca']
    #根据cfg对象中的参数生成一个训练数据加载器，并赋值
    train_loader = get_train_loader(cfg)
    #根据cfg对象中的参数生成一个模型对象，并赋值
    net = get_model(cfg)
    from model.model_tusimple import parsingNet1
    teacher_net = parsingNet1(pretrained = True, backbone='34', num_grid_row = 100, 
                             num_cls_row = 56, num_grid_col = 100, 
                             num_cls_col = 41, num_lane_on_row = 4, 
                             num_lane_on_col = 4, use_aux = False, 
                             input_height = 320, input_width = 800, 
                             fc_norm = False).cuda()


    #加载训练过的模型参数
    #读取保存好的预训练模型状态字典，并将其存储在resume_dict变量中
    resume_dict = torch.load('log/20240218_145649_lr_6e-04_b_1/model_026.pth', map_location='cpu')
    #将恢复的状态字典中的模型参数加载到目标模型
    teacher_net.load_state_dict(resume_dict['model'])    

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
    optimizer = get_optimizer(net, cfg)

    #判断cfg中是否有finetune这个参数，如果有，就表示要从一个预训练的模型开始微调，就执行下面的代码块
    #None
    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    #判断cfg中是否有resume这个参数，如果有，就表示要从一个之前保存的模型继续训练，就执行下面的代码块
    #None
    if cfg.resume is not None:
        #输出日志信息
        dist_print('==> Resume model from ' + cfg.resume)
        #读取保存好的预训练模型状态字典，并将其存储在resume_dict变量中
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        #将恢复的状态字典中的模型参数加载到目标模型
        net.load_state_dict(resume_dict['model'])
        #在每次恢复训练时，还需要将优化器之前保存的状态恢复回来
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        #将resume_epoch变量赋值为之前训练的轮数+1
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0
#创建一个学习率调度器，用来动态地调整学习率
    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    #分布式训练环境下打印训练数据集的长度
    dist_print(len(train_loader))
    #获取一个指标字典
    metric_dict = get_metric_dict(cfg)
    #根据配置信息cfg，调用get_loss_dict()函数来获取一个损失函数字典
    loss_dict = get_loss_dict(cfg)
    #创建日志记录器
    logger = get_logger(work_dir, cfg)
    # cp_projects(cfg.auto_backup, work_dir)
    #用于记录最优结果
    max_res = 0
    #用于保存评估结果
    res = None
    for epoch in range(resume_epoch, cfg.epoch):
        
        train(net, teacher_net,train_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, cfg.dataset)
        #调用训练数据加载器的reset()方法，重置加载器的状态，以便下一轮训练使用。
        train_loader.reset()
        #对神经网络模型进行评估，传入网络模型net、配置信息cfg、当前epoch值epoch以及日志记录器logger。
        #评估结果将保存在变量res中
        res = eval_lane(net, cfg, ep = epoch, logger = logger)

        if res is not None and res > max_res:
            #若评估结果更好，则将其赋值给max_res，以便跟踪记录最优结果。
            max_res = res
            #保存当前epoch对应的神经网络模型、优化器的状态
            save_model(net, optimizer, epoch, work_dir, distributed)
        #将最优结果max_res记录为一个标量值，并命名为'CuEval/X'。
        # 同时，使用参数global_step指定当前的全局步数为epoch
        logger.add_scalar('CuEval/X',max_res,global_step = epoch)
    #关闭日志记录器logger，完成日志的保存和关闭文件等操作
    logger.close()
