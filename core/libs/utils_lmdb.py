#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import io
import os
import random
from warnings import warn

import lmdb
import torch
import torchvision
import numpy as np

class LMDBEngine:
    def __init__(self, lmdb_path, write=False):
        self._write = write
        self._manual_close = False
        self._lmdb_path = lmdb_path
        if write and not os.path.exists(lmdb_path):
            os.makedirs(lmdb_path)
        if write:
            self._lmdb_env = lmdb.open(
                lmdb_path, map_size=1099511627776
            )
            self._lmdb_txn = self._lmdb_env.begin(write=True)
        else:
            self._lmdb_env = lmdb.open(
                lmdb_path, readonly=True, lock=False, readahead=False, meminit=True
            ) 
            self._lmdb_txn = self._lmdb_env.begin(write=False)
        # print('Load lmdb length:{}.'.format(len(self.keys())))

    def __getitem__(self, key_name):
        payload = self._lmdb_txn.get(key_name.encode())
        if payload is None:
            raise KeyError('Key:{} Not Found!'.format(key_name))
        try:
            image_buf = torch.tensor(np.frombuffer(payload, dtype=np.uint8))
            data = torchvision.io.decode_image(image_buf, mode=torchvision.io.ImageReadMode.RGB)
        except:
            data = torch.load(io.BytesIO(payload), weights_only=True)
        return data

    def __del__(self,):
        if not self._manual_close:
            warn('Writing engine not mannuly closed!', RuntimeWarning)
            self.close()

    def load(self, key_name, type='image', **kwargs):
        assert type in ['image', 'torch']
        payload = self._lmdb_txn.get(key_name.encode())
        if payload is None:
            raise KeyError('Key:{} Not Found!'.format(key_name))
        if type == 'torch':
            torch_data = torch.load(io.BytesIO(payload), weights_only=True)
            return torch_data
        elif type == 'image':
            image_buf = torch.tensor(np.frombuffer(payload, dtype=np.uint8))
            if 'mode' in kwargs.keys():
                if kwargs['mode'].lower() == 'rgb':
                    _mode = torchvision.io.ImageReadMode.RGB
                elif kwargs['mode'].lower() == 'rgba':
                    _mode = torchvision.io.ImageReadMode.RGB_ALPHA
                elif kwargs['mode'].lower() == 'gray':
                    _mode = torchvision.io.ImageReadMode.GRAY
                elif kwargs['mode'].lower() == 'graya':
                    _mode = torchvision.io.ImageReadMode.GRAY_ALPHA
                else:
                    raise NotImplementedError
            else:
                _mode = torchvision.io.ImageReadMode.RGB
            image_data = torchvision.io.decode_image(image_buf, mode=_mode)
            return image_data
        else:
            raise NotImplementedError

    def dump(self, key_name, payload, type='image', encode_jpeg=True):
        assert isinstance(payload, torch.Tensor) or isinstance(payload, dict), payload
        if not self._write:
            raise AssertionError('Engine Not Running in Write Mode.')
        if not hasattr(self, '_dump_counter'):
            self._dump_counter = 0
        assert type in ['image', 'torch']
        if self._lmdb_txn.get(key_name.encode()):
            print('Key:{} exsists!'.format(key_name))
            return 
        if type == 'torch':
            assert isinstance(payload, torch.Tensor) or isinstance(payload, dict), payload
            torch_buf = io.BytesIO()
            if isinstance(payload, torch.Tensor):
                torch.save(payload.detach().float().cpu(), torch_buf)
            else:
                for key in payload.keys():
                    payload[key] = payload[key].detach().float().cpu()
                torch.save(payload, torch_buf)
            payload_encoded = torch_buf.getvalue()
            # torch_data = torch.load(io.BytesIO(payload_encoded), weights_only=True)
            self._lmdb_txn.put(key_name.encode(), payload_encoded)
        elif type == 'image':
            assert payload.dim() == 3 and payload.shape[0] == 3
            if payload.max() < 2.0:
                print('Image Payload Should be [0, 255].')
            if encode_jpeg:
                payload_encoded = torchvision.io.encode_jpeg(payload.to(torch.uint8), quality=95)
            else:
                payload_encoded = torchvision.io.encode_png(payload.to(torch.uint8))
            payload_encoded = b''.join(map(lambda x:int.to_bytes(x,1,'little'), payload_encoded.numpy().tolist()))
            self._lmdb_txn.put(key_name.encode(), payload_encoded)
        else:
            raise NotImplementedError
        self._dump_counter += 1
        if self._dump_counter % 2000 == 0:
            self._lmdb_txn.commit()
            self._lmdb_txn = self._lmdb_env.begin(write=True)

    def exists(self, key_name):
        if self._lmdb_txn.get(key_name.encode()):
            return True
        else:
            return False

    def delete(self, key_name):
        if not self._write:
            raise AssertionError('Engine Not Running in Write Mode.')
        if not self.exists(key_name):
            print('Key:{} Not Found!'.format(key_name))
            return
        deleted = self._lmdb_txn.delete(key_name.encode())
        if not deleted:
            print('Delete Failed: {}!'.format(key_name))
            return
        self._lmdb_txn.commit()
        self._lmdb_txn = self._lmdb_env.begin(write=True)

    def raw_load(self, key_name, ):
        raw_payload = self._lmdb_txn.get(key_name.encode())
        return raw_payload

    def raw_dump(self, key_name, raw_payload):
        if not self._write:
            raise AssertionError('Engine Not Running in Write Mode.')
        if not hasattr(self, '_dump_counter'):
            self._dump_counter = 0
        self._lmdb_txn.put(key_name.encode(), raw_payload)
        self._dump_counter += 1
        if self._dump_counter % 2000 == 0:
            self._lmdb_txn.commit()
            self._lmdb_txn = self._lmdb_env.begin(write=True)

    def keys(self, ):
        all_keys = list(self._lmdb_txn.cursor().iternext(values=False))
        all_keys = [key.decode() for key in all_keys]
        # print('Found data, length:{}.'.format(len(all_keys)))
        return all_keys

    def close(self, ):
        if self._write:
            self._lmdb_txn.commit()
            self._lmdb_txn = self._lmdb_env.begin(write=True)
        self._lmdb_env.close()
        self._manual_close = True

    def random_visualize(self, vis_path, k=15, filter_key=None):
        all_keys = self.keys()
        if filter_key is not None:
            all_keys = [key for key in all_keys if filter_key in key]
        all_keys = random.choices(all_keys, k=k)
        print('Visualize: ', all_keys)
        images = [self.load(key, type='image')/255.0 for key in all_keys]
        images = [torchvision.transforms.functional.resize(i, (256, 256), antialias=True) for i in images]
        torchvision.utils.save_image(images, vis_path, nrow=5)
