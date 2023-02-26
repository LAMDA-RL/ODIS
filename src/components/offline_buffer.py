import re
import os
import torch as th
import numpy as np
import h5py
from sys import stderr

class OfflineSample():
    def __init__(self, data, batch_size, max_seq_length, device="cpu"):
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.data = data
        self.device = device
        for k, v in self.data.items():
            self.data[k] = v[:, :max_seq_length].to(self.device)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data:
                return self.data[item]
            elif hasattr(self, item):
                return getattr(self, item)
            else:
                raise ValueError('Cannot index OfflineSample with key "{}"'.format(item))
        else:
            raise ValueError('Cannot index OfflineSample with key "{}"'.format(item))

    def to(self, device):
        for k, v in self.data.items():
            self.data[k] = v.to(device)
        self.device = device

    def keys(self):
        return list(self.data.keys())


class OfflineBufferH5FullData():
    def __init__(self, datapath, offline_data_size=2000, device="cpu", random_sample=True):
        self.data = h5py.File(datapath, 'r')
        self.keys = list(self.data.keys())
        self.device = device
        original_buffer_size = self.data[self.keys[0]].shape[0]

        offline_data_size = original_buffer_size if offline_data_size > original_buffer_size or offline_data_size <= 0 else offline_data_size

        if random_sample:
            self.chosen_idx = np.random.choice(original_buffer_size, offline_data_size, replace=False)
        else:
            self.chosen_idx = np.array(range(original_buffer_size - offline_data_size, original_buffer_size))
        
        self.buffer_size = offline_data_size
        self.batch_size = self.buffer_size
        self.episodes_in_buffer = self.buffer_size
        
    def __del__(self):
        self.data.close()

    def max_t_filled(self, filled):
        return th.sum(filled, 1).max(0)[0]

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size
    
    def sample(self, batch_size):
        ep_ids = np.sort(np.random.choice(self.chosen_idx, batch_size, replace=False))
        episode_data = {k: th.tensor(self.data[k][ep_ids]) for k in self.keys}
        filled = episode_data['filled']
        max_ep_t = self.max_t_filled(filled).item()
        batch_sample = OfflineSample(episode_data, batch_size, max_ep_t, device=self.device)
        return batch_sample


class OfflineBufferH5():
    def __init__(self, datapaths, offline_data_size=2000, device="cpu", random_sample=True):
        offline_data_size = 100000000 if offline_data_size <= 0 else offline_data_size

        dataset_sources = len(datapaths)
        data_size_per_source = offline_data_size // dataset_sources
        dataset = [ self._read_data(datapaths[i], data_size_per_source, random_sample) for i in range(dataset_sources) ] 
        self.data = {
            k: np.concatenate([v[k] for v in dataset], axis=0) for k in dataset[0].keys()
        }

        self.keys = list(self.data.keys())
        self.buffer_size = self.data[self.keys[0]].shape[0]
        self.batch_size = self.buffer_size
        self.episodes_in_buffer = self.buffer_size

        self.device = device

    def _read_data(self, datapaths, offline_data_size, random_sample):
        data = {}
        for path in (datapaths):
            with h5py.File(path, 'r') as f:
                for k in f.keys():
                    if k not in data:
                        data[k] = f[k][:]
                    else:
                        data[k] = np.concatenate((data[k], f[k][:]), axis=0)
                    
            # pre-slice for memory releasing
            if not random_sample and data[list(data.keys())[0]].shape[0] > offline_data_size:
                data = {k: v[-offline_data_size:] for k, v in data.items()}

        keys = list(data.keys())
        original_buffer_size = len(data[keys[0]])
        offline_data_size = min(original_buffer_size, offline_data_size)
        
        if random_sample:
            idx = np.random.choice(original_buffer_size, offline_data_size, replace=False)
            data = {k: v[idx] for k, v in data.items()}
        
        return data

    def max_t_filled(self, filled):
        return th.sum(filled, 1).max(0)[0]

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size
    
    def sample(self, batch_size):
        ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
        episode_data = {k: th.tensor(v[ep_ids]) for k, v in self.data.items()}
        filled = episode_data['filled']
        max_ep_t = self.max_t_filled(filled).item()
        batch_sample = OfflineSample(episode_data, batch_size, max_ep_t, device=self.device)
        return batch_sample


class OfflineBufferPickle():
    def __init__(self, datapaths, offline_data_size=2000, device="cpu", random_sample=True):
        offline_data_size = 100000000 if offline_data_size <= 0 else offline_data_size

        dataset_sources = len(datapaths)
        data_size_per_source = offline_data_size // dataset_sources
        self.data = []
        for i in range(dataset_sources):
            self.data.extend(self._read_data(datapaths[i], data_size_per_source, random_sample))

        self.buffer_size = len(self.data)
        self.batch_size = self.buffer_size
        self.episodes_in_buffer = self.buffer_size
        self.keys = list(self.data[0].keys())
        self.device = device

    def _read_data(self, datapaths, offline_data_size, random_sample):
        data = []
        for path in (datapaths):
            data.extend(th.load(path))
            if not random_sample:
                if len(data) > offline_data_size:
                    data = data[-offline_data_size:]
        
        original_buffer_size = len(data)
        offline_data_size = min(original_buffer_size, offline_data_size)

        if random_sample:
            idx = np.random.choice(len(data), offline_data_size, replace=False)
            data = [data[i] for i in idx]
        return data

    def max_t_filled(self, filled):
        return th.sum(filled, 1).max(0)[0]

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
        episode_data = [self.data[i] for i in ep_ids]
        filled = th.cat([d['filled'].to(self.device) for d in episode_data], dim=0)
        max_ep_t = self.max_t_filled(filled).item()

        data = {
            k: th.cat( [d[k] for d in episode_data], dim=0 ) for k in self.keys
        }
        batch_sample = OfflineSample(data, batch_size, max_ep_t, device=self.device)
        return batch_sample


class OfflineBuffer():
    def __init__(self, map_name, quality, data_folder=None, dataset_folder='dataset', offline_data_size=2000, device="cpu", random_sample=True):
        datapaths = []
        if quality == 'medium-expert':
            datapaths.append(self._load_data_sources(dataset_folder, map_name, 'medium', data_folder))
            datapaths.append(self._load_data_sources(dataset_folder, map_name, 'expert', data_folder))
        else:
            datapaths.append(self._load_data_sources(dataset_folder, map_name, quality, data_folder))
        
        if all([all(['pkl' in f for f in paths]) for paths in datapaths]):
            self.buffer = OfflineBufferPickle(datapaths, offline_data_size=offline_data_size, device=device, random_sample=random_sample)
        elif all([all(['h5' in f for f in paths]) for paths in datapaths]):
            self.buffer = OfflineBufferH5(datapaths, offline_data_size=offline_data_size, device=device, random_sample=random_sample)
        else:
            raise ValueError("Cannot find parser for data files including {}".format(datapaths))

        self.buffer_size = self.buffer.buffer_size
        self.batch_size = self.buffer.buffer_size
        self.episodes_in_buffer = self.buffer.buffer_size
        self.device = device
    
    def _load_data_sources(self, dataset_folder, map_name, quality, data_folder):
        datapath = os.path.join(dataset_folder, map_name, quality)
        assert os.path.exists(datapath), "Offline data path {} does not exist".format(datapath)

        if data_folder is None or data_folder == '':
            existing_folders = [ f for f in sorted(os.listdir(datapath)) if os.path.isdir(os.path.join(datapath, f)) ]
            assert len(existing_folders) > 0
            data_folder = existing_folders[-1]
        
        dataset_path = os.path.join(datapath, data_folder)
        assert os.path.exists(dataset_path), 'Offline data path {} does not exist'.format(dataset_path)
        self.dataset_path = dataset_path
        print('Load dataset from {}'.format(dataset_path), file=stderr)

        filenames = os.listdir(dataset_path)
        if any(['part' in f for f in filenames]):
            datafiles = [f for f in filenames if 'part' in f]
            max_parts = max([ int(re.match(r'part_(\d+)\..*', file).group(1)) for file in datafiles if re.match(r'part_(\d+)\..*', file) is not None ])
            ext_name = os.path.splitext(datafiles[0])[1]
            datafiles = [ 'part_{}{}'.format(i, ext_name) for i in range(max_parts + 1) ]
        else:
            datafiles = filenames

        datapaths = [os.path.join(dataset_path, f) for f in datafiles]
        assert len(datapaths) > 0, 'dataset path {} contains no readable data files'.format(dataset_path)
        return datapaths

    def max_t_filled(self, filled):
        return self.buffer.max_t_filled(filled)

    def can_sample(self, batch_size):
        return self.buffer.can_sample(batch_size)

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)


class DataSaver():
    def __init__(self, datadir, max_size=2000):
        os.makedirs(datadir, exist_ok=True)
        self.datadir = datadir
        self.max_size = max_size
        self.data_batch = []
        self.part_no = 0

    def append(self, data):
        self.data_batch.append(data)
        if len(self.data_batch) >= self.max_size:
            self.save_batch()
            
    def save_batch(self):
        if len(self.data_batch) > 0:
            keys = list(self.data_batch[0].keys())
            datadic = {k: [] for k in keys}
            for d in self.data_batch:
                for k in keys:
                    if isinstance(d[k], th.Tensor):
                        datadic[k].append(d[k].numpy())
                    else:
                        datadic[k].append(d[k])
            datadic = {k: np.concatenate(v) for k, v in datadic.items()}

            with h5py.File(os.path.join(self.datadir, "part_{}.h5".format(self.part_no)), 'w') as file:
                for k, v in datadic.items():
                    file.create_dataset(k, data=v, compression='gzip', compression_opts=9)

            self.data_batch.clear()
            self.part_no += 1

    def close(self):
        self.save_batch()
        return self.datadir
