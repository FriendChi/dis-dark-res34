# Ultra-Fast-Lane-Detection-V2

 
# Ultra-Fast-Lane-Detection-V2

PyTorch实现的论文“https://arxiv.org/abs/2206.07389”。

 
PyTorch implementation of the paper "https://arxiv.org/abs/2206.07389".

! [] (ufldv2.png“能见度”)

 
![](ufldv2.png "vis")

#演示

 
# Demo

< a href = " https://youtu。

 
<a href="https://youtu.

be/VkvpoHlaMe0" target="_blank"><img src="http://img.youtube.com/vi/VkvpoHlaMe0/0.jpg" alt="Demo" width="240" height="180" /></a>

 
be/VkvpoHlaMe0" target="_blank"><img src="http://img.youtube.com/vi/VkvpoHlaMe0/0.jpg" alt="Demo" width="240" height="180" /></a>

#安装

 
# Install

请参见[INSTALL.md](./INSTALL.md)

 
Please see [INSTALL.md](./INSTALL.md)

#开始行动

 
# Get started

请在您想要运行的任何配置中修改' data_root '。

 
Please modify the `data_root` in any configs you would like to run.

我们将使用' configs/culane_res18.py '作为示例。

 
We will use `configs/culane_res18.py` as an example.

要训练模型，可以运行:

 
To train the model, you can run:

Python train.py configs/culane_res18.py——log_path /path/to/your/work/dir

 
python train.py configs/culane_res18.py --log_path /path/to/your/work/dir

或

 
or

Python -m torch. distribudit .launch——nproc_per_node=8 train.py configs/culane_res18.py——log_path /path/to/your/work/dir

 
python -m torch.distributed.launch --nproc_per_node=8 train.py configs/culane_res18.py --log_path /path/to/your/work/dir

需要注意的是，如果使用不同数量的gpu，学习率也要相应调整。

 
It should be noted that if you use different number of GPUs, the learning rate should be adjusted accordingly.

配置的学习率对应于CULane和CurveLanes数据集上的8个gpu训练。

 
The configs' learning rates correspond to 8-GPU training on CULane and CurveLanes datasets.

*如果你想用单个GPU在CULane或curvelane上进行训练，请将学习率降低1/8。

 
*If you want to train on CULane or CurveLanes with single GPU, please decrease the learning rate by a factor of 1/8.

*在tussimple上，学习率对应于单个GPU的训练。

 
* On the Tusimple, the learning rate corresponds to single GPU training.

#训练有素的模型

 
# Trained models

我们提供CULane, Tusimple和CurveLanes上的训练模型。

 
We provide trained models on CULane, Tusimple, and CurveLanes.

开始使用

 
开始使用

“data_root”。

 
请修改你想要运行的任何配置文件中的 data_root。

我们将以配置/ culane_res18.py为例。

 
我们将以 configs/culane_res18.py 为例。

要训练模型，你可以运行：

 
要训练模型，你可以运行：

Python train.py configs/culane_res18.py——log_path /path/to/your/work/dir

 
python train.py configs/culane_res18.py --log_path /path/to/your/work/dir

python G:/postgraduate_studyfile/Ultra-Fast-Lane-Detection-v2-master/train.py configs/tusimple_res18.py——log_path G:/postgraduate_studyfile/Ultra-Fast-Lane-Detection-v2-master/model_data .py

 
python G:/postgraduate_studyfile/Ultra-Fast-Lane-Detection-v2-master/train.py configs/tusimple_res18.py --log_path G:/postgraduate_studyfile/Ultra-Fast-Lane-Detection-v2-master/model_data

或者

 
或者

Python -m torch. distribudit .launch——nproc_per_node=8 train.py configs/culane_res18.py——log_path /path/to/your/work/dir

 
python -m torch.distributed.launch --nproc_per_node=8 train.py configs/culane_res18.py --log_path /path/to/your/work/dir

(2)图形处理器，图形处理器，图形处理器，图形处理器。

 
需要注意的是，如果你使用不同数量的GPU，学习率应该相应地调整。

8 .图形处理器，图形处理器

 
配置文件中的学习率对应于在CULane和CurveLanes数据集上使用8个GPU进行训练。

* 1/8。

 
*如果你想在CULane或CurveLanes上使用单个GPU进行训练，请将学习率减少1/8倍。

* * * * * *

 
*在Tusimple上，学习率对应于单个GPU训练。

训练好的模型

 
训练好的模型

曲线图，曲线图，曲线图。

 
我们提供了在CULane，Tusimple和CurveLanes上训练好的模型。

| Dataset | Backbone | F1 | Link |

 
| Dataset | Backbone | F1 | Link |

|------------|----------|-------|------|

 
|------------|----------|-------|------|

| CULane | ResNet18 | 75.0 | https://drive.google.com/file/d/1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W/view?usp=sharing/https://pan.baidu.com/s/1Z3W4y3eA9xrXJ51-voK4WQ?pwd=pdzs |

 
| CULane | ResNet18 | 75.0 | https://drive.google.com/file/d/1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W/view?usp=sharing/https://pan.baidu.com/s/1Z3W4y3eA9xrXJ51-voK4WQ?pwd=pdzs |

| CULane | ResNet34 | 76.0 | https://drive.google.com/file/d/1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj/view?usp=sharing/https://pan.baidu.com/s/1PHNpVHboQlmpjM5NXl9IxQ?pwd=jw8f |

 
| CULane | ResNet34 | 76.0 | https://drive.google.com/file/d/1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj/view?usp=sharing/https://pan.baidu.com/s/1PHNpVHboQlmpjM5NXl9IxQ?pwd=jw8f |

|图森| ResNet18 | 96.11 | https://drive.google.com/file/d/1Clnj9-dLz81S3wXiYtlkc4HVusCb978t/view?usp=sharing/https://pan.baidu.com/s/1umHo0RZIAQ1l_FzL2aZomw?pwd=6xs1 |

 
| Tusimple | ResNet18 | 96.11 | https://drive.google.com/file/d/1Clnj9-dLz81S3wXiYtlkc4HVusCb978t/view?usp=sharing/https://pan.baidu.com/s/1umHo0RZIAQ1l_FzL2aZomw?pwd=6xs1 |

|图森| ResNet34 | 96.24 | https://drive.google.com/file/d/1pkz8homK433z39uStGK3ZWkDXrnBAMmX/view?usp=sharing/https://pan.baidu.com/s/1Eq7oxnDoE0vcQGzs1VsGZQ?pwd=b88p |

 
| Tusimple | ResNet34 | 96.24 | https://drive.google.com/file/d/1pkz8homK433z39uStGK3ZWkDXrnBAMmX/view?usp=sharing/https://pan.baidu.com/s/1Eq7oxnDoE0vcQGzs1VsGZQ?pwd=b88p |

| CurveLanes | ResNet18 | 80.42 | https://drive.google.com/file/d/1VfbUvorKKMG4tUePNbLYPp63axgd-8BX/view?usp=sharing/https://pan.baidu.com/s/1jCqKqgSQdh6nwC5pYpYO1A?pwd=urhe |

 
| CurveLanes | ResNet18 | 80.42 | https://drive.google.com/file/d/1VfbUvorKKMG4tUePNbLYPp63axgd-8BX/view?usp=sharing/https://pan.baidu.com/s/1jCqKqgSQdh6nwC5pYpYO1A?pwd=urhe |

| CurveLanes | ResNet34 | 81.34 | https://drive.google.com/file/d/1O1kPSr85Icl2JbwV3RBlxWZYhLEHo8EN/view?usp=sharing/https://pan.baidu.com/s/1fk2Wg-1QoHXTnTlasSM6uQ?pwd=4mn3 |

 
| CurveLanes | ResNet34 | 81.34 | https://drive.google.com/file/d/1O1kPSr85Icl2JbwV3RBlxWZYhLEHo8EN/view?usp=sharing/https://pan.baidu.com/s/1fk2Wg-1QoHXTnTlasSM6uQ?pwd=4mn3 |

要进行评估，请运行

 
For evaluation, run

壳牌

 
Shell

mkdir tmp

 
mkdir tmp

Python test.py configs/culane_res18.py——test_model /path/to/your/model.pth——test_work_dir ./tmp

 
python test.py configs/culane_res18.py --test_model /path/to/your/model.pth --test_work_dir ./tmp

与训练相同，还支持多gpu评估。

 
Same as training, multi-gpu evaluation is also supported.

壳牌

 
Shell

mkdir tmp

 
mkdir tmp

——test_model /path/to/your/model.pth——test_work_dir ./tmp

 
python -m torch.distributed.launch --nproc_per_node=8 test.py configs/culane_res18.py --test_model /path/to/your/model.pth --test_work_dir ./tmp

#可视化

 
# Visualization

我们提供了一个脚本来可视化检测结果。

 
We provide a script to visualize the detection results.

运行以下命令以可视化CULane的测试集。

 
Run the following commands to visualize on the testing set of CULane.

Python demo.py configs/culane_res18.py——test_model /path/to/your/culane_res18.pth

 
python demo.py configs/culane_res18.py --test_model /path/to/your/culane_res18.pth

# tensort部署

 
# Tensorrt Deploy

我们还提供了一个python脚本来对视频进行张排序推理。

 
We also provide a python script to do tensorrt inference on videos.

1.

 
1.

转换为onnx模型

 
Convert to onnx model

Python deploy/pt2onnx.py——config_path configs/culane_res34.py——model_path weights/culane_res34.pth

 
python deploy/pt2onnx.py --config_path configs/culane_res34.py --model_path weights/culane_res34.pth

也可以通过以下脚本下载onnx模型:https://github.com/PINTO0309/PINTO_model_zoo/blob/main/324_Ultra-Fast-Lane-Detection-v2/download.sh。

 
Or you can download the onnx model using the following script: https://github.com/PINTO0309/PINTO_model_zoo/blob/main/324_Ultra-Fast-Lane-Detection-v2/download.sh.

并复制' ufldv2_culane_res34_320x1600。

 
And copy `ufldv2_culane_res34_320x1600.

到' weights/ufldv2_culane_res34_320x1600.onnx '

 
onnx` to `weights/ufldv2_culane_res34_320x1600.onnx`

2.

 
2.

转换为张排序模型

 
Convert to tensorrt model

使用trtexec转换引擎模型

 
Use trtexec to convert engine model

“trtexec——onnx =重量/ culane_res34。

 
`trtexec --onnx=weights/culane_res34.

onnx——saveEngine =重量/ culane_res34.engine '

 
onnx --saveEngine=weights/culane_res34.engine`

3.

 
3.

做推理

 
Do inference

——config_path configs/culane_res34.py——engine_path weights/culane_res34. py

 
python deploy/trt_infer.py --config_path configs/culane_res34.py --engine_path weights/culane_res34.

引擎——video_path example.mp4

 
engine --video_path example.mp4

#引用

 
# Citation

助理

 
BibTeX

@InProceedings {qin2020ultra,

 
@InProceedings{qin2020ultra,

作者={秦，泽群，王，环宇，李，喜}，

 
author = {Qin, Zequn and Wang, Huanyu and Li, Xi},

title ={超快速结构感知深车道检测}，

 
title = {Ultra Fast Structure-aware Deep Lane Detection},

{欧洲计算机视觉会议(ECCV)}，

 
booktitle = {The European Conference on Computer Vision (ECCV)},

年份= {2020}

 
year = {2020}

}

 
}

@ARTICLE {qin2022ultrav2,

 
@ARTICLE{qin2022ultrav2,

作者={秦，泽群，张，彭义，李，喜}，

 
author={Qin, Zequn and Zhang, Pengyi and Li, Xi},

{IEEE模式分析与机器智能汇刊}，

 
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},

title={基于混合锚驱动有序分类的超快速深巷检测};

 
title={Ultra Fast Deep Lane Detection With Hybrid Anchor Driven Ordinal Classification},

年= {2022},

 
year={2022},

体积= {},

 
volume={},

数量= {},

 
number={},

页面= {1 - 14},

 
pages={1-14},

doi = {10.1109 / TPAMI.2022。

 
doi={10.1109/TPAMI.2022

# Ultra-Fast-Lane-Detection-V2
PyTorch implementation of the paper "[Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification](https://arxiv.org/abs/2206.07389)".


![](ufldv2.png "vis")

# Demo 
<a href="https://youtu.be/VkvpoHlaMe0
" target="_blank"><img src="http://img.youtube.com/vi/VkvpoHlaMe0/0.jpg" 
alt="Demo" width="240" height="180" border="10" /></a>


# Install
Please see [INSTALL.md](./INSTALL.md)

# Get started
Please modify the `data_root` in any configs you would like to run. We will use `configs/culane_res18.py` as an example.

To train the model, you can run:
```
python train.py configs/culane_res18.py --log_path /path/to/your/work/dir
```
or
```
python -m torch.distributed.launch --nproc_per_node=8 train.py configs/culane_res18.py --log_path /path/to/your/work/dir
```
It should be noted that if you use different number of GPUs, the learning rate should be adjusted accordingly. The configs' learning rates correspond to 8-GPU training on CULane and CurveLanes datasets. **If you want to train on CULane or CurveLanes with single GPU, please decrease the learning rate by a factor of 1/8.** On the Tusimple, the learning rate corresponds to single GPU training.
# Trained models
We provide trained models on CULane, Tusimple, and CurveLanes.

开始使用
请修改你想要运行的任何配置文件中的 data_root。我们将以 configs/culane_res18.py 为例。

要训练模型，你可以运行：

python train.py configs/culane_res18.py --log_path /path/to/your/work/dir
python G:/postgraduate_studyfile/Ultra-Fast-Lane-Detection-v2-master/train.py configs/tusimple_res18.py --log_path G:/postgraduate_studyfile/Ultra-Fast-Lane-Detection-v2-master/model_data

或者

python -m torch.distributed.launch --nproc_per_node=8 train.py configs/culane_res18.py --log_path /path/to/your/work/dir

需要注意的是，如果你使用不同数量的GPU，学习率应该相应地调整。配置文件中的学习率对应于在CULane和CurveLanes数据集上使用8个GPU进行训练。*如果你想在CULane或CurveLanes上使用单个GPU进行训练，请将学习率减少1/8倍。*在Tusimple上，学习率对应于单个GPU训练。

训练好的模型
我们提供了在CULane，Tusimple和CurveLanes上训练好的模型。

| Dataset    | Backbone | F1   | Link |
|------------|----------|-------|------|
| CULane     | ResNet18 | 75.0  |  [Google](https://drive.google.com/file/d/1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1Z3W4y3eA9xrXJ51-voK4WQ?pwd=pdzs)    |
| CULane     | ResNet34 | 76.0  |   [Google](https://drive.google.com/file/d/1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1PHNpVHboQlmpjM5NXl9IxQ?pwd=jw8f)   |
| Tusimple   | ResNet18 | 96.11 |   [Google](https://drive.google.com/file/d/1Clnj9-dLz81S3wXiYtlkc4HVusCb978t/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1umHo0RZIAQ1l_FzL2aZomw?pwd=6xs1)   |
| Tusimple   | ResNet34 | 96.24 |   [Google](https://drive.google.com/file/d/1pkz8homK433z39uStGK3ZWkDXrnBAMmX/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1Eq7oxnDoE0vcQGzs1VsGZQ?pwd=b88p)   |
| CurveLanes | ResNet18 | 80.42 |   [Google](https://drive.google.com/file/d/1VfbUvorKKMG4tUePNbLYPp63axgd-8BX/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1jCqKqgSQdh6nwC5pYpYO1A?pwd=urhe)   |
| CurveLanes | ResNet34 | 81.34 |   [Google](https://drive.google.com/file/d/1O1kPSr85Icl2JbwV3RBlxWZYhLEHo8EN/view?usp=sharing)/[Baidu](https://pan.baidu.com/s/1fk2Wg-1QoHXTnTlasSM6uQ?pwd=4mn3)   |

For evaluation, run
```Shell
mkdir tmp

python test.py configs/culane_res18.py --test_model /path/to/your/model.pth --test_work_dir ./tmp
```

Same as training, multi-gpu evaluation is also supported.
```Shell
mkdir tmp

python -m torch.distributed.launch --nproc_per_node=8 test.py configs/culane_res18.py --test_model /path/to/your/model.pth --test_work_dir ./tmp
```

# Visualization
We provide a script to visualize the detection results. Run the following commands to visualize on the testing set of CULane.
```
python demo.py configs/culane_res18.py --test_model /path/to/your/culane_res18.pth
```

# Tensorrt Deploy
We also provide a python script to do tensorrt inference on videos.

1. Convert to onnx model
    ```
    python deploy/pt2onnx.py --config_path configs/culane_res34.py --model_path weights/culane_res34.pth
    ```
    Or you can download the onnx model using the following script: https://github.com/PINTO0309/PINTO_model_zoo/blob/main/324_Ultra-Fast-Lane-Detection-v2/download.sh. And copy `ufldv2_culane_res34_320x1600.onnx` to `weights/ufldv2_culane_res34_320x1600.onnx`

2. Convert to tensorrt model

    Use trtexec to convert engine model

    `trtexec --onnx=weights/culane_res34.onnx --saveEngine=weights/culane_res34.engine`

3. Do inference
    ```
    python deploy/trt_infer.py --config_path  configs/culane_res34.py --engine_path weights/culane_res34.engine --video_path example.mp4
    ```

# Citation

```BibTeX
@InProceedings{qin2020ultra,
author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
title = {Ultra Fast Structure-aware Deep Lane Detection},
booktitle = {The European Conference on Computer Vision (ECCV)},
year = {2020}
}

@ARTICLE{qin2022ultrav2,
  author={Qin, Zequn and Zhang, Pengyi and Li, Xi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Ultra Fast Deep Lane Detection With Hybrid Anchor Driven Ordinal Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TPAMI.2022.3182097}
}
```
