import os
import random
import torch, math
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq
import numpy as np
import torchvision
import torch.nn.functional as F

from PIL import Image
import io, gc

from dataset.utils import DegradationProcessor


def decode_image(image_data):
    return Image.open(io.BytesIO(image_data)).convert('RGB')


class PromptFixDataset(IterableDataset):
    def __init__(self, data_dir, resolution, seed=42):
        self.data_dir = data_dir
        self.resolution = resolution
        self.flip_prob = 0.5
        self.seed = seed

        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.parquet')
        ])
        
        self.total_len = 0
        for file_path in self.files:
            # Get the number of rows in the Parquet file
            parquet_file = pq.ParquetFile(file_path)
            num_rows = parquet_file.metadata.num_rows
            self.total_len += num_rows
            
        self.degradation_processor = DegradationProcessor()

    def pil_to_tensor(self, inp_img, tar_img):
        width, height = inp_img.size
        aspect_ratio = float(width) / float(height)
        if width < height:
            new_width = self.resolution
            new_height = int(self.resolution / aspect_ratio)
        else:
            new_height = self.resolution
            new_width = int(self.resolution * aspect_ratio)
        inp_img = inp_img.resize((new_width, new_height), Image.LANCZOS)
        tar_img = tar_img.resize((new_width, new_height), Image.LANCZOS)

        inp_img = np.array(inp_img).astype(np.float32).transpose(2, 0, 1)
        inp_img_tensor = torch.tensor((inp_img / 127.5 - 1.0).astype(np.float32))
        tar_img = np.array(tar_img).astype(np.float32).transpose(2, 0, 1)
        tar_img_tensor = torch.tensor((tar_img / 127.5 - 1.0).astype(np.float32))
        crop = torchvision.transforms.RandomCrop(self.resolution)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((inp_img_tensor, tar_img_tensor)))).chunk(2)
        return image_0, image_1

    def __iter__(self):
        # Obtain rank and world_size
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        total_files = len(self.files)

        total_workers = total_workers
        files_per_worker = math.ceil(total_files / total_workers)

        files_assigned = []
        files_assigned_index = []

        for i in range(files_per_worker):
            file_index = (global_worker_id + i * total_workers) % total_files
            files_assigned.append(self.files[file_index])
            files_assigned_index.append(file_index)

        # Shuffle the files assigned to this worker
        random_seed = self.seed + global_worker_id
        random.seed(random_seed)
        random.shuffle(files_assigned)
        
        print(f"Global Worker ID: {global_worker_id}, Assigned Files: {len(files_assigned)}, Files Index: {files_assigned_index}")
        
        if len(files_assigned) == 0:
            print(f"Worker {global_worker_id} has no files assigned.")

        for file_path in files_assigned:
            parquet_file = pq.ParquetFile(file_path)
            num_row_groups = parquet_file.num_row_groups

            # Optionally shuffle row groups
            row_group_indices = list(range(num_row_groups))
            random.shuffle(row_group_indices)

            for row_group_idx in row_group_indices:
                batch = parquet_file.read_row_group(row_group_idx)
                df = batch.to_pandas()

                # Optionally shuffle the DataFrame
                df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

                for _, row in df.iterrows():
                    input_img = decode_image(row['input_img']['bytes'])
                    processed_img = decode_image(row['processed_img']['bytes'])
                    input_img, processed_img = self.pil_to_tensor(input_img, processed_img)
                    instruction = row['instruction']
                    auxiliary_prompt = row['auxiliary_prompt']
                        
                    # if random.random() < 0.1:
                    #     LQ, _, HQ_USM = self.degradation_processor.feed_data(processed_img.unsqueeze(0)*0.5+0.5)
                    #     LQ, HQ_USM = LQ[0]*2-1, HQ_USM[0]*2-1
                    #     processed_img = HQ_USM
                    #     input_img = LQ
                    #     if row['task_id'] != 'Deblur':
                    #         auxiliary_prompt = ""
                    #     instruction = random.choice(self.sr_inst_prompts)
                    #     pass

                    yield {
                        'edited': processed_img,
                        'edit': {
                            'c_concat': input_img,
                            'c_crossattn': [instruction, auxiliary_prompt]
                        }
                    }
            
            del parquet_file
            gc.collect()