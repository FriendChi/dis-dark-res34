#安装
 # Install
1.  1. 克隆项目
  Clone the project

“‘壳
 ```Shell
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection-V2
 git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection-V2
cd Ultra-Fast-Lane-Detection-V2
 cd Ultra-Fast-Lane-Detection-V2
' ' '
 ```

2.  2. 创建一个conda虚拟环境并激活它
  Create a conda virtual environment and activate it

“‘壳
 ```Shell
Conda create -n lane-det python=3.7 -y
 conda create -n lane-det python=3.7 -y
Conda启动航道探测
 conda activate lane-det
' ' '
 ```

3. 3. 安装依赖关系
  Install dependencies

“‘壳
 ```Shell
如果你没有pytorch
 # If you dont have pytorch
pytorch-cuda=11.7 -c pytorch-c nvidia
 conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

PIP install -r requirements.txt
 pip install -r requirements.txt

PIP安装——extra-index-url https://developer.download.nvidia.com/compute/redist——升级nvidia-dali-cuda110
 pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
安装Nvidia DALI(非常快的数据加载库)
 # Install Nvidia DALI (Very fast data loading lib))

cd my_interp
 cd my_interp

sh build.sh
 sh build.sh
#如果失败，你可能需要升级你的GCC到v7.3.0
 # If this fails, you might need to upgrade your GCC to v7.3.0
' ' '
 ```

4.  4. 数据准备
  Data preparation
#### **4.1 Tusimple数据集**
 #### **4.1 Tusimple dataset**
下载[CULane](https://xingangpan.github.io/projects/CULane.html)、[Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3)或[CurveLanes](https://github.com/SoulmateB/CurveLanes)。 Download [CULane](https://xingangpan.github.io/projects/CULane.html), [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3), or [CurveLanes](https://github.com/SoulmateB/CurveLanes) as you want. Tusimple的目录安排应该类似于(' test_label。  The directory arrangement of Tusimple should look like(`test_label.json '可以从[这里](https://github.com/TuSimple/tusimple-benchmark/issues/3)下载:
  json` can be downloaded from [here](https://github.com/TuSimple/tusimple-benchmark/issues/3) ):
' ' '
 ```
TUSIMPLE美元
 $TUSIMPLE
|──剪辑
 |──clips
|──label_data_0313.json
 |──label_data_0313.json
|──label_data_0531.json
 |──label_data_0531.json
|──label_data_0601.json
 |──label_data_0601.json
|──test_tasks_0627.json
 |──test_tasks_0627.json
|──test_label.json
 |──test_label.json
|──readme.md
 |──readme.md
' ' '
 ```
对于Tusimple，没有提供分割注释，因此我们需要从json注释中生成分割。
 For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation.

“‘壳
 ```Shell
Python脚本/convert_tusimple.py——root /path/to/your/tusimple
 python scripts/convert_tusimple.py --root /path/to/your/tusimple

#这将生成分段和两个列表文件:train_gt.txt和test.txt
 # this will generate segmentations and two list files: train_gt.txt and test.txt
' ' '
 ```
#### **4.2 CULane数据集**
 #### **4.2 CULane dataset**
CULane的目录安排应该是这样的:
 The directory arrangement of CULane should look like:
' ' '
 ```
CULANE美元
 $CULANE
|──driver_100_30frame
 |──driver_100_30frame
|──driver_161_90frame
 |──driver_161_90frame
|──driver_182_30frame
 |──driver_182_30frame
|──driver_193_90frame
 |──driver_193_90frame
|──driver_23_30frame
 |──driver_23_30frame
|──driver_37_30frame
 |──driver_37_30frame
|──laneseg_label_w16
 |──laneseg_label_w16
|──列表
 |──list
' ' '
 ```
对于CULane，请运行:
 For CULane, please run:
“‘壳
 ```Shell
Python脚本/cache_culane_ponits.py——root /path/to/your/culane
 python scripts/cache_culane_ponits.py --root /path/to/your/culane

#这将生成一个culane_anno_cache。 # this will generate a culane_anno_cache.包含所有车道注释的Json文件，可用于加速训练而无需读取车道分割图
  json file containing all the lane annotations, which can be used for speed up training without reading lane segmentation maps
' ' '
 ```
#### **4.3 CurveLanes数据集**
 #### **4.3 CurveLanes dataset**
CurveLanes的目录安排应该如下:
 The directory arrangement of CurveLanes should look like:
' ' '
 ```
CurveLanes美元
 $CurveLanes
|──测试
 |──test
|──火车
 |──train
|──有效
 |──valid
' ' '
 ```
对于CurveLanes，请运行:
 For CurveLanes, please run:
“‘壳
 ```Shell
Python scripts/convert_curvelanes.py——root /path/to/your/curvelanes
 python scripts/convert_curvelanes.py --root /path/to/your/curvelanes

Python脚本/make_curvelane_as_culane_test.py——root /path/to/your/curvelanes
 python scripts/make_curvelane_as_culane_test.py --root /path/to/your/curvelanes

#这也将生成一个curvelanes_anno_cache_train。 # this will also generate a curvelanes_anno_cache_train.json文件。  json file. 此外，将在val集上生成许多.lines.txt文件，以启用CULane风格评估。
  Moreover, many .lines.txt file will be generated on the val set to enable CULane style evaluation.
' ' '
 ```

5.  5. 安装CULane评估工具(仅用于测试)。
  Install CULane evaluation tools (Only required for testing).

如果你只是想训练一个模型或做一个演示，这个工具是不必要的，你可以跳过这一步。 If you just want to train a model or make a demo, this tool is not necessary and you can skip this step. 如果你想在CULane上得到评估结果，你应该安装这个工具。
  If you want to get the evaluation results on CULane, you should install this tool.

该工具需要opencvc++。 This tools requires OpenCV C++. 请按照[这里](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)安装OpenCV c++。  Please follow [here](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) to install OpenCV C++. ***当您构建OpenCV时，请从PATH中删除anaconda的路径，否则将失败
  ***When you build OpenCV, remove the paths of anaconda from PATH or it will be failed.***
“‘壳
 ```Shell
#首先你需要安装OpenCV c++。
 # First you need to install OpenCV C++.
安装完成后，制作一个OpenCV包含路径的软链接。
 # After installation, make a soft link of OpenCV include path.

-s /usr/local/include/opencv2 /usr/local/include/opencv2
 ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2
' ' '
 ```
我们提供了三种复杂的管道来构建CULane的评价工具。
 We provide three kinds of complie pipelines to build the evaluation tool of CULane.

选项1:
 Option 1:

“‘壳
 ```Shell
cd / culane评价
 cd evaluation/culane
使
 make
' ' '
 ```

选项2:
 Option 2:
“‘壳
 ```Shell
cd / culane评价
 cd evaluation/culane
Mkdir build && CD build
 mkdir build && cd build
cmake . .
 cmake ..
使
 make
Mv culane_evaluator ../evaluate
 mv culane_evaluator ../evaluate
' ' '
 ```

Windows用户:
 For Windows user:
“‘壳
 ```Shell
mkdir build-vs2017
 mkdir build-vs2017
cd build-vs2017
 cd build-vs2017
cmake . . cmake .. -G Visual Studio 15 2017 Win64
  -G "Visual Studio 15 2017 Win64"
make——build。 cmake --build . ——配置版本
  --config Release
#或者，打开“xxx. exe”文件。 # or, open the "xxx.然后单击build按钮
  sln" file by Visual Studio and click build button
移动culane_evaluator ../evaluate move culane_evaluator ../evaluate


# Install
1. Clone the project

    ```Shell
    git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection-V2
    cd Ultra-Fast-Lane-Detection-V2
    ```

2. Create a conda virtual environment and activate it

    ```Shell
    conda create -n lane-det python=3.7 -y
    conda activate lane-det
    ```

3. Install dependencies

    ```Shell
    # If you dont have pytorch
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

    pip install -r requirements.txt

    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
    # Install Nvidia DALI (Very fast data loading lib))

    cd my_interp

    sh build.sh
    # If this fails, you might need to upgrade your GCC to v7.3.0
    ```

4. Data preparation
    #### **4.1 Tusimple dataset**
    Download [CULane](https://xingangpan.github.io/projects/CULane.html), [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3), or [CurveLanes](https://github.com/SoulmateB/CurveLanes) as you want. The directory arrangement of Tusimple should look like(`test_label.json` can be downloaded from [here](https://github.com/TuSimple/tusimple-benchmark/issues/3) ):
    ```
    $TUSIMPLE
    |──clips
    |──label_data_0313.json
    |──label_data_0531.json
    |──label_data_0601.json
    |──test_tasks_0627.json
    |──test_label.json
    |──readme.md
    ```
    For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

    ```Shell
    python scripts/convert_tusimple.py --root /path/to/your/tusimple

    # this will generate segmentations and two list files: train_gt.txt and test.txt
    ```
    #### **4.2 CULane dataset**
    The directory arrangement of CULane should look like:
    ```
    $CULANE
    |──driver_100_30frame
    |──driver_161_90frame
    |──driver_182_30frame
    |──driver_193_90frame
    |──driver_23_30frame
    |──driver_37_30frame
    |──laneseg_label_w16
    |──list
    ```
    For CULane, please run:
    ```Shell
    python scripts/cache_culane_ponits.py --root /path/to/your/culane

    # this will generate a culane_anno_cache.json file containing all the lane annotations, which can be used for speed up training without reading lane segmentation maps
    ```
    #### **4.3 CurveLanes dataset**
    The directory arrangement of CurveLanes should look like:
    ```
    $CurveLanes
    |──test
    |──train
    |──valid
    ```
    For CurveLanes, please run:
    ```Shell
    python scripts/convert_curvelanes.py --root /path/to/your/curvelanes

    python scripts/make_curvelane_as_culane_test.py --root /path/to/your/curvelanes

    # this will also generate a curvelanes_anno_cache_train.json file. Moreover, many .lines.txt file will be generated on the val set to enable CULane style evaluation.
    ```

5. Install CULane evaluation tools (Only required for testing). 

    If you just want to train a model or make a demo, this tool is not necessary and you can skip this step. If you want to get the evaluation results on CULane, you should install this tool.

    This tools requires OpenCV C++. Please follow [here](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) to install OpenCV C++. ***When you build OpenCV, remove the paths of anaconda from PATH or it will be failed.***
    ```Shell
    # First you need to install OpenCV C++. 
    # After installation, make a soft link of OpenCV include path.

    ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2
    ```
    We provide three kinds of complie pipelines to build the evaluation tool of CULane.

    Option 1:

    ```Shell
    cd evaluation/culane
    make
    ```

    Option 2:
    ```Shell
    cd evaluation/culane
    mkdir build && cd build
    cmake ..
    make
    mv culane_evaluator ../evaluate
    ```

    For Windows user:
    ```Shell
    mkdir build-vs2017
    cd build-vs2017
    cmake .. -G "Visual Studio 15 2017 Win64"
    cmake --build . --config Release  
    # or, open the "xxx.sln" file by Visual Studio and click build button
    move culane_evaluator ../evaluate
    ```
