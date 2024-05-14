import torch, os
from utils.common import merge_config, get_model
from evaluation.eval_wrapper import eval_lane
import torch
if __name__ == "__main__":
    #启用 cudnn 的自动优化，根据输入的大小和类型选择最合适的算法，可以提高运行速度
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    #判断是否使用分布式训练
    # 如果是的话，就设置当前进程使用的 GPU 设备，并初始化通信组
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    cfg.distributed = distributed
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    net = get_model(cfg)
    #加载预训练的模型参数，并将其赋值给模型对象
    state_dict = torch.load(cfg.test_model, map_location = 'cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict = True)
    #判断是否使用分布式训练，
    # 如果是的话，就将模型对象包装成一个分布式数据并行的模块，使得不同的 GPU 设备可以协同工作
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
    #判断是否存在测试结果保存的目录，如果不存在，就创建一个
    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)
    #调用评估函数
    eval_lane(net, cfg)