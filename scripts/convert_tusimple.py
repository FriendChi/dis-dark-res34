import os
import cv2
import tqdm
import numpy as np
import pdb
import json, argparse

#calc_k函数：用来计算车道线的方向，即斜率。
#输入一条车道的坐标
def calc_k(line):
    '''
    Calculate the direction of lanes
    '''
    #提取所有坐标
    line_x = line[::2]
    line_y = line[1::2]
    #计算车道的像素长度
    length = np.sqrt((line_x[0]-line_x[-1])**2 + (line_y[0]-line_y[-1])**2)
    if length < 90:
        return -10                                          # if the lane is too short, it will be skipped
    ## 对x和y坐标进行一次最小二乘法的多项式拟合
    # n表示多项式的次数。p是一个长度为n+1的向量，表示多项式的系数，按照降幂排列。
    p = np.polyfit(line_x, line_y,deg = 1)
    # 计算斜率对应的弧度值，使用反正切函数
    rad = np.arctan(p[0])
    
    return rad
#用来在分割标签上绘制车道线，并且按照从左到右的顺序分别用1,2,3,4表示
def draw(im,line,idx,show = False):
    '''
    Generate the segmentation label according to json annotation
    '''
    line_x = line[::2]
    line_y = line[1::2]
    #表示线条起点
    pt0 = (int(line_x[0]),int(line_y[0]))
    if show:
        #在图像im上绘制文字，文字内容为idx参数的字符串形式 文字位置为线条中点的上方20个像素处
        #文字字体为cv2.FONT_HERSHEY_SIMPLEX，文字大小为1.0，文字颜色为白色（255, 255, 255），文字线型为cv2.LINE_AA
        cv2.putText(im,str(idx),(int(line_x[len(line_x) // 2]),int(line_y[len(line_x) // 2]) - 20),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        #表示线条的颜色值
        
    idx = idx * 60    
    #遍历line_x列表中除了最后一个元素之外的所有元素的索引
    for i in range(len(line_x)-1):
        #坐标连线成车道线,颜色太浅会导致肉眼无法观察
        #cv2.line(im,pt0,(int(line_x[i+1]),int(line_y[i+1])),255,thickness = 16)
        cv2.line(im,pt0,(int(line_x[i+1]),int(line_y[i+1])),(idx,),thickness = 16)
        pt0 = (int(line_x[i+1]),int(line_y[i+1]))
#用来从json文件中读取Tusimple数据集的文件名、车道线坐标和水平采样点。
def get_tusimple_list(root, label_list):
    '''
    Get all the files' names from the json annotation
    '''
    label_json_all = []
    for l in label_list:
        #获取文件夹下json文件路径
        l = os.path.join(root,l)
        # json.loads 函数对列表中的每个字符串进行解析，将其转换为 Python 的字典对象
        label_json = [json.loads(line) for line in open(l).readlines()]
        #label_json_all有三个大列表，每个列表代表一个json
        label_json_all += label_json
    #提取key为raw_file的值
    names = [l['raw_file'] for l in label_json_all]
    h_samples = [np.array(l['h_samples']) for l in label_json_all]
    lanes = [np.array(l['lanes']) for l in label_json_all]
    #raw_file：表示图片的文件名，例如 "raw_file": "clips/0313-1/6040/20.jpg"。
    #h_samples：水平采样点的列表，即图片每行的 y 坐标，例如 "h_samples": [240, 250, 260...]
    #lanes：表示车道线坐标的列表，即图片中每条车道线上每个水平采样点对应的 x 坐标，
    #例如 "lanes": [[-2, -2, -2, -2, 632, 625, 617, 609], [-2, -2, -2, -2, 719, 734, 748, 761], 
    # [-2, -2, -2, -2, -2, -2, -2, 890], [367, 374, 381, 389, 397, 405, 412, 418]]
    
    
    line_txt = []
    #len(lanes)输出子列表个数
    for i in range(len(lanes)):
        line_txt_i = []
        #取出x坐标
        for j in range(len(lanes[i])):
            #坐标为 -2，则表示该车道线在该水平采样点处不存在或不可见
            #查询该列表是否全为-2
            if np.all(lanes[i][j] == -2):
                continue
            valid = lanes[i][j] != -2
            #定义一个长度为 (len(h_samples[i][valid])+len(lanes[i][j][valid]))) 的空列表
            #使用了布尔数组索引,i代表图像，j代表对应水平线，valid代表布尔
            line_txt_tmp = [None]*(len(h_samples[i][valid])+len(lanes[i][j][valid]))
            #每两个元素表达一个坐标点
            line_txt_tmp[::2] = list(map(str,lanes[i][j][valid]))
            line_txt_tmp[1::2] = list(map(str,h_samples[i][valid]))
            line_txt_i.append(line_txt_tmp)
        #line_txt第一维度是图像文件，第二维度是车道线，第三维度是坐标
        line_txt.append(line_txt_i)

    return names,line_txt
#用来根据车道线坐标生成分割标签，并且生成训练列表，同时保存车道线坐标的缓存文件。
#传入路径，车道坐标，文件名
def generate_segmentation_and_train_list(root, line_txt, names):
    """
    The lane annotations of the Tusimple dataset is not strictly in order, so we need to find out the correct lane order for segmentation.
    We use the same definition as CULane, in which the four lanes from left to right are represented as 1,2,3,4 in segentation label respectively.
    """
    train_gt_fp = open(os.path.join(root,'train_gt.txt'),'w')
    
    cache_dict = {}

    for i in tqdm.tqdm(range(len(line_txt))):
        #逐个读取图像文件
        tmp_line = line_txt[i]
        lines = []
        for j in range(len(tmp_line)):
            #读取文件的车道线
            #一个车道线成一列表
            #map 可以对一个可迭代对象中的每个元素应用一个函数，并返回一个新的可迭代对象
            lines.append(list(map(float,tmp_line[j])))
        #计算每条车道的拟合的曲率
        ks = np.array([calc_k(line) for line in lines])             # get the direction of each lane
        #筛选负值，即左侧车道并复制
        k_neg = ks[ks<0].copy()
        k_pos = ks[ks>0].copy()
        #去掉-10这个车道过短的值
        k_neg = k_neg[k_neg != -10]                                      # -10 means the lane is too short and is discarded
        k_pos = k_pos[k_pos != -10]
        #从左到右对车道进行排序
        k_neg.sort()
        k_pos.sort()
        #生成第i个文件的标签文件名
        label_path = names[i][:-3]+'png'
        #建立全零矩阵
        label = np.zeros((720,1280),dtype=np.uint8)
        #4个车道，56个水平线，xy坐标
        all_points = np.zeros((4,56,2), dtype=np.float)
        #56个水平线位置
        the_anno_row_anchor = np.array([160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
       270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370,
       380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480,
       490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590,
       600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700,
       710])
        #将y坐标进行赋值
        all_points[:,:,1] = np.tile(the_anno_row_anchor, (4,1))
        #赋值x进行初始化
        all_points[:,:,0] = -99999
        #表示车道线是否存在，0表示不在
        bin_label = [0,0,0,0]
        #左侧是否只有一条车道线
        if len(k_neg) == 1:  
            #where 根据一个条件，返回数组中满足该条件的元素的索引      
            # 找到 k_neg[0] 对应的车道线的索引。k_neg[0] 是左侧最靠近中心线的（在只有一条车道线的情况下）                           # for only one lane in the left
            which_lane = np.where(ks == k_neg[0])[0][0]
            #在label空矩阵上将坐标连线成车道线
            draw(label,lines[which_lane],2)
            xx = np.array(lines[which_lane][::2])
            #这是为了将 y 坐标映射到 the_anno_row_anchor 数组中对应的位置
            yy = ((np.array(lines[which_lane][1::2]) - 160) / 10).astype(int)
            #形状为 (4,56,2)
            all_points[1,yy,0] = xx
            #确定这条车道线存在
            bin_label[1] = 1
        elif len(k_neg) == 2:                                         # for two lanes in the left
            which_lane = np.where(ks == k_neg[1])[0][0]
            draw(label,lines[which_lane],1)
            xx = np.array(lines[which_lane][::2])
            yy = ((np.array(lines[which_lane][1::2]) - 160) / 10).astype(int)
            all_points[0,yy,0] = xx
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label,lines[which_lane],2)
            xx = np.array(lines[which_lane][::2])
            yy = ((np.array(lines[which_lane][1::2]) - 160) / 10).astype(int)
            all_points[1,yy,0] = xx
            bin_label[0] = 1
            bin_label[1] = 1
            #多于两条车道，只选择最近两条车道
        elif len(k_neg) > 2:                                           # for more than two lanes in the left, 
            which_lane = np.where(ks == k_neg[1])[0][0]                # we only choose the two lanes that are closest to the center
            draw(label,lines[which_lane],1)
            xx = np.array(lines[which_lane][::2])
            yy = ((np.array(lines[which_lane][1::2]) - 160) / 10).astype(int)
            all_points[0,yy,0] = xx
            which_lane = np.where(ks == k_neg[0])[0][0]
            draw(label,lines[which_lane],2)
            xx = np.array(lines[which_lane][::2])
            yy = ((np.array(lines[which_lane][1::2]) - 160) / 10).astype(int)
            all_points[1,yy,0] = xx
            bin_label[0] = 1
            bin_label[1] = 1

        if len(k_pos) == 1:                                            # For the lanes in the right, the same logical is adopted.
            which_lane = np.where(ks == k_pos[0])[0][0]
            draw(label,lines[which_lane],3)
            xx = np.array(lines[which_lane][::2])
            yy = ((np.array(lines[which_lane][1::2]) - 160) / 10).astype(int)
            all_points[2,yy,0] = xx
            bin_label[2] = 1
        elif len(k_pos) == 2:
            which_lane = np.where(ks == k_pos[1])[0][0]
            draw(label,lines[which_lane],3)
            xx = np.array(lines[which_lane][::2])
            yy = ((np.array(lines[which_lane][1::2]) - 160) / 10).astype(int)
            all_points[2,yy,0] = xx
            which_lane = np.where(ks == k_pos[0])[0][0]
            draw(label,lines[which_lane],4)
            xx = np.array(lines[which_lane][::2])
            yy = ((np.array(lines[which_lane][1::2]) - 160) / 10).astype(int)
            all_points[3,yy,0] = xx
            bin_label[2] = 1
            bin_label[3] = 1
        elif len(k_pos) > 2:
            which_lane = np.where(ks == k_pos[-1])[0][0]
            draw(label,lines[which_lane],3)
            xx = np.array(lines[which_lane][::2])
            yy = ((np.array(lines[which_lane][1::2]) - 160) / 10).astype(int)
            all_points[2,yy,0] = xx
            which_lane = np.where(ks == k_pos[-2])[0][0]
            draw(label,lines[which_lane],4)
            xx = np.array(lines[which_lane][::2])
            yy = ((np.array(lines[which_lane][1::2]) - 160) / 10).astype(int)
            all_points[3,yy,0] = xx
            bin_label[2] = 1
            bin_label[3] = 1
        #将标签图像label保存到指定的路径中，路径由root和label_path两个变量拼接而成
        cv2.imwrite(os.path.join(root,label_path),label)
        #将all_points数组转换为列表，并赋值给cache_dict字典中以names[i]为键的值
        cache_dict[names[i]] = all_points.tolist()
        #向训练列表文件中写入一行内容，内容由names[i]（图像文件名），label_path（标签文件名），以及bin_label（车道线存在性）三个变量组成，
        # 中间用空格分隔，最后加上换行符
        train_gt_fp.write(names[i] + ' ' + label_path + ' '+' '.join(list(map(str,bin_label))) + '\n')
    train_gt_fp.close()
    with open(os.path.join(root, 'tusimple_anno_cache.json'), 'w') as f:
        #将cache_dict字典中的数据转换为json格式，并写入到f文件
        json.dump(cache_dict, f)
#get_args函数：用来解析命令行参数，主要是Tusimple数据集的根目录。
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the Tusimple dataset')
    return parser

#root=r"G:\postgraduate_studyfile\Tusimple\train_set"
#root = '/mnt/studyfile/Tusimple/train_set'
#产生了两个txt文件和一个json文件，多个png图片
if __name__ == "__main__":
    #args = get_args().parse_args()

    # training set
    names,line_txt = get_tusimple_list("/kaggle/input/tusimple/TUSimple/train_set",  ['label_data_0601.json','label_data_0531.json','label_data_0313.json'])
    # generate segmentation and training list for training
    generate_segmentation_and_train_list("/kaggle/working/", line_txt, names)

    # testing set
    names,line_txt = get_tusimple_list("/kaggle/input/tusimple/TUSimple/test_set", ['test_tasks_0627.json'])
    # generate testing set for testing
    with open(os.path.join("/kaggle/working/",'test.txt'),'w') as fp:
        for name in names:
            fp.write(name + '\n')

