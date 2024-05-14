import torch
import numpy as np
import random
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import json
import os
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy
import my_interp
class LaneExternalIterator(object):
    def __init__(self, path, list_path, batch_size=None, shard_id=None, num_shards=None, mode = 'train', dataset_name=None):
        assert mode in ['train', 'test']
        self.mode = mode
        self.path = path
        self.list_path = list_path
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards

        if isinstance(list_path, str):
            with open(list_path, 'r') as f:
                total_list = f.readlines()
        elif isinstance(list_path, list) or isinstance(list_path, tuple):
            total_list = []
            for lst_path in list_path:
                with open(lst_path, 'r') as f:
                    total_list.extend(f.readlines())
        else:
            raise NotImplementedError
        if self.mode == 'train':
            if dataset_name == 'CULane':
                cache_path = os.path.join(path, 'culane_anno_cache.json')
            elif dataset_name == 'Tusimple':
                cache_path = os.path.join("/kaggle/working/", 'tusimple_anno_cache.json')
            elif dataset_name == 'CurveLanes':
                cache_path = os.path.join(path, 'train', 'curvelanes_anno_cache.json')
            else:
                raise NotImplementedError

            if shard_id == 0:
                print('loading cached data')
            cache_fp = open(cache_path, 'r')
            self.cached_points = json.load(cache_fp)
            if shard_id == 0:
                print('cached data loaded')

        self.total_len = len(total_list)
    
        self.list = total_list[self.total_len * shard_id // num_shards:
                                self.total_len * (shard_id + 1) // num_shards]
        self.n = len(self.list)

    def __iter__(self):
        self.i = 0
        if self.mode == 'train':
            random.shuffle(self.list)
        return self

    def _prepare_train_batch(self):
        images = []
        seg_images = []
        labels = []

        for _ in range(self.batch_size):
            l = self.list[self.i % self.n]
            l_info = l.split()
            img_name = l_info[0]
            seg_name = l_info[1]

            if img_name[0] == '/':
                img_name = img_name[1:]
            if seg_name[0] == '/':
                seg_name = seg_name[1:]
                
            img_name = img_name.strip()
            seg_name = seg_name.strip()
            
            img_path = os.path.join(self.path, img_name)
            with open(img_path, 'rb') as f:
                images.append(np.frombuffer(f.read(), dtype=np.uint8))

            img_path = os.path.join(self.path, seg_name)
            with open(img_path, 'rb') as f:
                seg_images.append(np.frombuffer(f.read(), dtype=np.uint8))

            points = np.array(self.cached_points[img_name])
            labels.append(points.astype(np.float32))

            self.i = self.i + 1
            
        return (images, seg_images, labels)

    
    def _prepare_test_batch(self):
        images = []
        names = []
        for _ in range(self.batch_size):
            img_name = self.list[self.i % self.n].split()[0]

            if img_name[0] == '/':
                img_name = img_name[1:]
            img_name = img_name.strip()

            img_path = os.path.join(self.path, img_name)

            with open(img_path, 'rb') as f:
                images.append(np.frombuffer(f.read(), dtype=np.uint8))
            names.append(np.array(list(map(ord,img_name))))
            self.i = self.i + 1
            
        return images, names

    def __next__(self):
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration
        if self.mode == 'train':
            res = self._prepare_train_batch()
        elif self.mode == 'test':
            res = self._prepare_test_batch()
        else:
            raise NotImplementedError

        return res
    def __len__(self):
        return self.total_len

    next = __next__

def encoded_images_sizes(jpegs):
    shapes = fn.peek_image_shape(jpegs)  # the shapes are HWC
    h = fn.slice(shapes, 0, 1, axes=[0]) # extract height...
    w = fn.slice(shapes, 1, 1, axes=[0]) # ...and width...
    return fn.cat(w, h)               # ...and concatenate

def ExternalSourceTrainPipeline(batch_size, num_threads, device_id, external_data, train_width, train_height, top_crop, normalize_image_scale = False, nscale_w = None, nscale_h = None):
    #创建一个Pipeline对象，指定批次大小、线程数量和设备ID
    pipe = Pipeline(batch_size, num_threads, device_id)
    #with语句来定义管道的作用域
    with pipe:
        #从external_data这个迭代器中获取三个输出：
        # jpegs，seg_images和labels。
        # jpegs是JPEG编码的图像，seg_images是图像的分割掩码，labels是图像中车道线的坐标。
        jpegs, seg_images, labels = fn.external_source(source=external_data, num_outputs=3)
        #对jpegs和seg_images进行解码，得到RGB格式的图像。
        # 指定device参数为"mixed"，表示在GPU上执行解码操作。
        images = fn.decoders.image(jpegs, device="mixed")
        seg_images = fn.decoders.image(seg_images, device="mixed")
        #为false
        if normalize_image_scale:
            #使所有图像的尺寸一致
            images = fn.resize(images, resize_x=nscale_w, resize_y=nscale_h)
            seg_images = fn.resize(seg_images, resize_x=nscale_w, resize_y=nscale_h, interp_type=types.INTERP_NN)
            # make all images at the same size
        #获取jpegs的尺寸，并计算中心点
        size = encoded_images_sizes(jpegs)
        center = size / 2
        #创建一个仿射变换矩阵mt，用于对图像和标签进行随机的缩放、旋转和平移变换
        #使用fn.random.uniform操作符来生成随机的缩放比例、旋转角度和平移偏移量
        mt = fn.transforms.scale(scale = fn.random.uniform(range=(0.8, 1.2), shape=[2]), center = center)
        mt = fn.transforms.rotation(mt, angle = fn.random.uniform(range=(-6, 6)), center = center)

        off = fn.cat(fn.random.uniform(range=(-200, 200), shape = [1]), fn.random.uniform(range=(-100, 100), shape = [1]))
        mt = fn.transforms.translation(mt, offset = off)
        #对images和seg_images进行仿射变换，根据mt矩阵进行变形
        images = fn.warp_affine(images, matrix = mt, fill_value=0, inverse_map=False)
        seg_images = fn.warp_affine(seg_images, matrix = mt, fill_value=0, inverse_map=False)
        labels = fn.coord_transform(labels.gpu(), MT = mt)


        images = fn.resize(images, resize_x=train_width, resize_y=int(train_height/top_crop))
        seg_images = fn.resize(seg_images, resize_x=train_width, resize_y=int(train_height/top_crop), interp_type=types.INTERP_NN)


        images = fn.crop_mirror_normalize(images, 
                                            dtype=types.FLOAT, 
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        seg_images = fn.crop_mirror_normalize(seg_images, 
                                            dtype=types.FLOAT, 
                                            mean = [0., 0., 0.],
                                            std = [1., 1., 1.],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        pipe.set_outputs(images, seg_images, labels)
    return pipe

def ExternalSourceValPipeline(batch_size, num_threads, device_id, external_data, train_width, train_height):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, labels = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(jpegs, device="mixed")
        images = fn.resize(images, resize_x=train_width, resize_y=int(train_height/0.6)+1)
        images = fn.crop_mirror_normalize(images, 
                                            dtype=types.FLOAT, 
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        pipe.set_outputs(images, labels.gpu())
    return pipe

def ExternalSourceTestPipeline(batch_size, num_threads, device_id, external_data):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, names = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(jpegs, device="mixed")

        images = fn.resize(images, resize_x=800, resize_y=288)
        images = fn.crop_mirror_normalize(images, 
                                            dtype=types.FLOAT, 
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255])

        names = fn.pad(names, axes=0, fill_value = -1, shape = 46)
        pipe.set_outputs(images, names)
    return pipe
# from data.constant import culane_row_anchor, culane_col_anchor
#同时具有TrainLoader和DataSet的功能
class TrainCollect:
    def __init__(self, batch_size, num_threads, data_root, list_path, shard_id, num_shards, row_anchor, col_anchor, train_width, train_height, num_cell_row, num_cell_col,
    dataset_name, top_crop):
        #从数据根目录和列表路径中读取数据
        eii = LaneExternalIterator(data_root, list_path, batch_size=batch_size, shard_id=shard_id, num_shards=num_shards, dataset_name = dataset_name)
        #确定图片大小
        if dataset_name == 'CULane':
            self.original_image_width = 1640
            self.original_image_height = 590
        elif dataset_name == 'Tusimple':
            self.original_image_width = 1280
            self.original_image_height = 720
        elif dataset_name == 'CurveLanes':
            self.original_image_width = 2560
            self.original_image_height = 1440

        if dataset_name == 'CurveLanes':
            pipe = ExternalSourceTrainPipeline(batch_size, num_threads, shard_id, eii, train_width, train_height,top_crop, normalize_image_scale = True, nscale_w = 2560, nscale_h = 1440)
        else:
            #num_threads是占据内存的工作进程数，shard_id: 一个整数，表示当前进程的ID，eii: 一个LaneExternalInputIterator的对象，用于从数据集中读取和转换数据
            #train_width和train_height是训练图像的宽高，top_crop：裁剪图像时保留的顶部比例
            pipe = ExternalSourceTrainPipeline(batch_size, num_threads, shard_id, eii, train_width, train_height,top_crop)
        self.pii = DALIGenericIterator(pipe, output_map = ['images', 'seg_images', 'points'], last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)
        self.eii_n = eii.n
        self.batch_size = batch_size

        self.interp_loc_row = torch.tensor(row_anchor, dtype=torch.float32).cuda() * self.original_image_height
        self.interp_loc_col = torch.tensor(col_anchor, dtype=torch.float32).cuda() * self.original_image_width
        self.num_cell_row = num_cell_row
        self.num_cell_col = num_cell_col

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.pii)
        images = data[0]['images']
        seg_images = data[0]['seg_images']
        points = data[0]['points']
        points_row = my_interp.run(points, self.interp_loc_row, 0)
        points_row_extend = self._extend(points_row[:,:,:,0]).transpose(1,2)
        labels_row = (points_row_extend / self.original_image_width * (self.num_cell_row - 1)).long()
        labels_row[points_row_extend < 0] = -1
        labels_row[points_row_extend > self.original_image_width] = -1
        labels_row[labels_row < 0] = -1
        labels_row[labels_row > (self.num_cell_row - 1)] = -1

        points_col = my_interp.run(points, self.interp_loc_col, 1)
        points_col = points_col[:,:,:,1].transpose(1,2)
        labels_col = (points_col / self.original_image_height * (self.num_cell_col - 1)).long()
        labels_col[points_col < 0] = -1
        labels_col[points_col > self.original_image_height] = -1
        
        labels_col[labels_col < 0] = -1
        labels_col[labels_col > (self.num_cell_col - 1)] = -1

        labels_row_float = points_row_extend / self.original_image_width
        labels_row_float[labels_row_float<0] = -1
        labels_row_float[labels_row_float>1] = -1

        labels_col_float = points_col / self.original_image_height
        labels_col_float[labels_col_float<0] = -1
        labels_col_float[labels_col_float>1] = -1

        return {'images':images, 'seg_images':seg_images, 'labels_row':labels_row, 'labels_col':labels_col, 'labels_row_float':labels_row_float, 'labels_col_float':labels_col_float}
    
    def __len__(self):
        return int((self.eii_n + self.batch_size - 1) / self.batch_size)
    def reset(self):
        self.pii.reset()
    next = __next__

    def _extend(self, coords):
        # coords : n x num_lane x num_cls
        n, num_lanes, num_cls = coords.shape
        coords_np = coords.cpu().numpy()
        coords_axis = np.arange(num_cls)
        fitted_coords = coords.clone()
        for i in range(n):
            for j in range(num_lanes):
                lane = coords_np[i,j]
                if lane[-1] > 0:
                    continue

                valid = lane > 0
                num_valid_pts = np.sum(valid)
                if num_valid_pts < 6:
                    continue
                p = np.polyfit(coords_axis[valid][num_valid_pts//2:], lane[valid][num_valid_pts//2:], deg = 1)   
                start_point = coords_axis[valid][num_valid_pts//2]
                fitted_lane = np.polyval(p, np.arange(start_point, num_cls))

                
                fitted_coords[i,j,start_point:] = torch.tensor(fitted_lane, device = coords.device)
        return fitted_coords
    def _extend_col(self, coords):
        pass


class TestCollect:
    def __init__(self, batch_size, num_threads, data_root, list_path, shard_id, num_shards):
        self.batch_size = batch_size
        eii = LaneExternalIterator(data_root, list_path, batch_size=batch_size, shard_id=shard_id, num_shards=num_shards, mode = 'test')
        pipe = ExternalSourceTestPipeline(batch_size, num_threads, shard_id, eii)
        self.pii = DALIGenericIterator(pipe, output_map = ['images', 'names'], last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)
        self.eii_n = eii.n
    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.pii)
        images = data[0]['images']
        names = data[0]['names']
        restored_names = []
        for name in names:
            if name[-1] == -1:
                restored_name = ''.join(list(map(chr,name[:-1])))
            else:
                restored_name = ''.join(list(map(chr,name)))
            restored_names.append(restored_name)
            
        out_dict = {'images': images, 'names': restored_names}
        return out_dict
    
    def __len__(self):
        return int((self.eii_n + self.batch_size - 1) / self.batch_size)

    def reset(self):
        self.pii.reset()
    next = __next__


