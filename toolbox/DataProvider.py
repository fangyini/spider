import numpy as np
import math
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SuperResolutionProvider(object):
    """
    provider for mobile traffic data (or other 3D data)
    ------------------------------------------------------
    :arg
    input_size: 3-element tuple(x*y*feature map), the shape of input we want
    output_size: 2-element tuple (x*y), the shape of output we want
    prediction_gap: int, the distant between the last input frame and output frame
    flatten: bool, whether flatten the output or not
    batchsize: int, the size of batch, default -1 (take all data generated)
    stride: 2-element tuple, the stride when selecting data
    shuffle: bool, default True. shuffle the data or not
    pad: 2-element or None, the size of padding, default None
    pad_value: float, padding values


    """

    def __init__(self, input_size, output_size, batchsize=-1, stride=(1, 1),
                 shuffle=True):

        self.input_size = input_size
        self.output_size = output_size
        self.batchsize = batchsize
        self.stride = stride
        self.shuffle = shuffle
        y = []
        for i in range(0, 1008, 144):
            filename = './models/training_with_selection/selec_' + str(i) + '_' + str(i + 144) + '.pt'
            try:
                y_i = torch.stack(torch.load(filename, map_location=torch.device('cpu'))).to(device)
                y.append(y_i)
            except:
                print('no file, t=' + str(i))
                continue
        for i in range(0, 432, 144):
            filename = './models/training_with_selection/selec_' + str(i) + '_' + str(i + 144) + '.pt'
            try:
                y_i = torch.stack(torch.load(filename, map_location=torch.device('cpu'))).to(device)
                y.append(y_i)
            except:
                print('no file, t=' + str(i))
                continue
        self.y = torch.stack(y).view(-1, 100, 100).permute(1, 2, 0).cpu().numpy()

    def DataSlicer_3D(self, inputs, excerpt):
        """
        generate data from input frames
        ------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        excerpt: list, the index of start frame of inputs
        flatten: bool, flatten target
        external: np.array (x*y*t), target from another resource, should has the shape (x'*y'*t)

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y)
        external_data: np.array, with dim (batchsize*1)
        """

        x_max, y_max, z_max = inputs.shape
        x_num = int(math.ceil((x_max - self.input_size[0] + 1.0) / self.stride[0]))
        y_num = int(math.ceil((y_max - self.input_size[1] + 1.0) / self.stride[1]))
        total = x_num * y_num * len(excerpt)
        if self.batchsize <= 0:
            self.batchsize = total
        input_data = np.zeros((self.input_size[0], self.input_size[1], self.input_size[2], total))

        target_data = np.zeros((self.output_size[0], self.output_size[1], total))

        data_num = 0

        for frame in range(len(excerpt)):

            input_frame = inputs[:, :, excerpt[frame]:excerpt[frame] + self.input_size[2]]
            target_frame = inputs[:, :, excerpt[frame] + self.input_size[2] - 1]

            for x in range(self.input_size[0], x_max + 1, self.stride[0]):
                for y in range(self.input_size[1], y_max + 1, self.stride[1]):
                    input_data[:, :, :, data_num] = input_frame[x - self.input_size[0]:x, y - self.input_size[1]:y, :]
                    target_data[:, :, data_num] = target_frame[x - self.input_size[0]:
                        x - self.input_size[0] + self.output_size[0], y - self.input_size[1]:
                            y - self.input_size[1] + self.output_size[1]]

                    data_num += 1

        if self.shuffle:
            indices = np.random.permutation(total)
            return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                    np.transpose(target_data[:, :, indices[0:self.batchsize]], (2, 0, 1)).reshape(self.batchsize, -1))

        else:
            return (np.transpose(input_data[:, :, :, 0:self.batchsize], (3, 2, 0, 1)),
                    np.transpose(target_data[:, :, 0:self.batchsize], (2, 0, 1)).reshape(self.batchsize, -1))

    def SparseDataGenerator(self, inputs):
        points_number = self.input_size[0] * self.input_size[1]
        for x in range(1440):
            input_frame = inputs[:, :, x].copy()
            rate = float(random.randint(10, 90))
            print('Rate ' + str(rate))
            sampled = int(points_number * rate / 100)
            mask = np.array([0] * (points_number - sampled) + [1] * sampled)
            np.random.shuffle(mask)
            mask = np.reshape(mask, (self.input_size[0], self.input_size[1]))
            inputs[:, :, x] = np.multiply(input_frame, mask)
        np.save('milan_test_sparse', inputs)
        print('Finished')

    # inputs: (100, 100, 5760) the entire dataset, excerpt: the starting index
    # outputs: sparse net_inputs (100, 6, 100, 100), ground truth net_targets (100, 10000)
    # self.input_size (100, 100, 6)
    # self.output_size (100, 100)
    def DataProcessor(self, inputs, sparse_inputs, excerpts, isTrain):
        net_inputs = np.zeros((self.batchsize, self.input_size[0], self.input_size[1], self.input_size[2])) # (100, 100, 100, 6)
        net_targets = np.zeros((self.batchsize, self.output_size[0], self.output_size[1])) # (100, 100, 100)
        #if isTrain:
        points_number = self.input_size[0] * self.input_size[1] * self.input_size[2]
        for ind in range(self.batchsize):
            starting = excerpts[ind]
            temp = inputs[:, :, starting:(starting+self.input_size[2])].copy()
            net_targets[ind, :, :] = temp[:, :, self.input_size[2] - 1]
            '''rate = float(random.randint(10, 70))
            sampled = int(points_number * rate / 100)
            mask = np.array([0] * (points_number - sampled) + [1] * sampled)
            np.random.shuffle(mask)
            mask = np.reshape(mask, (temp.shape))'''
            mod_ind = starting%1440
            mask = self.y[:, :, mod_ind:(mod_ind+self.input_size[2])]
            mask_ind = np.where(mask == 0)
            temp[mask_ind] = 0
            # unknown = 0
            #temp = np.multiply(temp, mask)
            net_inputs[ind, :, :, :] = temp
        '''else:
            for ind in range(self.batchsize-self.input_size[2]):
                starting = excerpts[ind]
                temp = inputs[:, :, starting:(starting + self.input_size[2])].copy()
                temp_in = sparse_inputs[:, :, starting:(starting+self.input_size[2])].copy()
                #print('starting='+str(starting)+', ending='+str(starting+self.input_size[2]))
                #temp_in[np.nonzero(temp_in)] = (temp_in[np.nonzero(temp_in)] - mean) / std
                net_inputs[ind, :, :, :] = temp_in
                net_targets[ind, :, :] = temp[:, :, self.input_size[2]-1]'''
        net_targets = np.reshape(net_targets, (self.batchsize, self.output_size[0]*self.output_size[1]))
        net_inputs = np.transpose(net_inputs, (0, 3, 1, 2))
        return net_inputs, net_targets

    def maskGenerator(self):
        masks = np.zeros((self.input_size[0], self.input_size[1], self.input_size[2]))
        points_number = self.input_size[0] * self.input_size[1]
        for x in range(self.input_size[2]):
            rate = float(random.randint(10, 90))
            #print('Rate ' + str(rate))
            sampled = int(points_number * rate / 100)
            mask = np.array([0] * (points_number - sampled) + [1] * sampled)
            np.random.shuffle(mask)
            mask = np.reshape(mask, (self.input_size[0], self.input_size[1]))
            masks[:, :, x] = mask
        return masks  # (100, 100, 6)

    def feed(self, inputs, sparse_inputs, isTrain):
        """
        iterate over mini-batch
        --------------------------------------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        framebatch: int, maximum frames we selected in one mini-batch
        mean: float, inputs normalized constant, mean
        std: float, inputs normalized constant, standard error
        norm_tar: bool, target normalized as input, default False

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y) or flatten one
        """

        frame_max = inputs.shape[2] - self.input_size[2] + 1

        if self.shuffle:
            indices = np.random.permutation(frame_max)

        #for start_idx in range(0, frame_max, framebatch):
        for start_idx in range(0, frame_max-self.batchsize, self.batchsize):
            '''if self.shuffle:
                excerpt = indices[start_idx:start_idx + framebatch]
            else:
                excerpt = range(start_idx, min((start_idx + framebatch), frame_max))
            net_inputs, net_targets = self.DataSlicer_3D(inputs=inputs, excerpt=excerpt)'''

            if self.shuffle:
                excerpts = indices[start_idx:(start_idx+self.batchsize)]
            else:
                excerpts = [start_idx]
            # net_inputs (100, 6, 100, 100), net_targets (100, 10000), excerpt is the starting index
            net_inputs, net_targets = self.DataProcessor(inputs=inputs, sparse_inputs=sparse_inputs, excerpts=excerpts, isTrain=isTrain)
            yield net_inputs, net_targets, excerpts
            '''if norm_tar:
                net_targets = ((net_targets - mean) / float(std))
            yield (net_inputs - mean) / float(std), net_targets'''


def norm(data):
    m = np.mean(data)
    s = np.mean(data)
    data = (data - m) / s
    return data


class MoverProvider(object):
    """
    provider for data in 2 files
    ------------------------------------------------------
    :arg
    batchsize: int, the size of batch, default -1 (take all data generated)
    shuffle: bool, default True. shuffle the data or not
    """

    def __init__(self, length):
        self.length = length

    def feed(self, inputs, rate, special=False, keepdims=False):

        """
        generate data from input files
        ------------------------------------------------
        :arg
        inputs: np.array, the source input data
        targets: np.array, the source target data

        :return
        input: np.array
        target: np.array
        """

        np.random.seed(1)
        end = inputs.shape[-1] - self.length + 1
        #mask = np.random.choice([0, 1], size=inputs.shape, p=[1 - rate, rate])
        y = []
        for i in range(0, 1008, 144):
            filename = './models/training_with_selection/selec_' + str(i) + '_' + str(i + 144) + '.pt'
            try:
                y_i = torch.stack(torch.load(filename, map_location=torch.device('cpu'))).to(device)
                y.append(y_i)
            except:
                print('no file, t=' + str(i))
                continue
        for i in range(0, 432, 144):
            filename = './models/training_with_selection/selec_' + str(i) + '_' + str(i + 144) + '.pt'
            try:
                y_i = torch.stack(torch.load(filename, map_location=torch.device('cpu'))).to(device)
                y.append(y_i)
            except:
                print('no file, t=' + str(i))
                continue
        mask = torch.stack(y).view(-1, 100, 100).permute(1, 2, 0).cpu().numpy()
        sparse = np.multiply(mask, inputs)

        for start_idx in range(end):
            excerpt = range(start_idx, start_idx + self.length)
            input_fra = inputs[:, :, excerpt]
            #mean = np.mean(input_fra)
            #std = np.std(input_fra)
            out_x = np.transpose(sparse[:, :, excerpt], (2, 0, 1))
            #out_x[np.nonzero(out_x)] = (out_x[np.nonzero(out_x)] - mean) / std
            out_x = np.expand_dims(out_x, axis=0)
            out_y = inputs[:, :, start_idx+self.length-1]
            out_y = np.reshape(out_y, (1, out_y.shape[0]*out_y.shape[1]))
            yield out_x, out_y
