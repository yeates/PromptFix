# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
from torch import inf


def load_checkpoint(file_name, config, model, model_ema, optimizer, lr_scheduler, loss_scaler, logger):
    if config.model.params.deepspeed != '':
        file_name = file_name.split('/')
        ckptdir = '/'.join(file_name[:-1])
        tag = file_name[-1]
        _, client_states = model.load_checkpoint(ckptdir, tag=tag)
        print(client_states)
        logger.info("Resume checkpoint %s" % file_name)
        checkpoint = torch.load(
            os.path.join(ckptdir, tag, "state.pth"), map_location="cpu"
        )
        msg = model_ema.load_state_dict(checkpoint['model_ema'])
        logger.info(msg)
        start_epoch = checkpoint["epoch"] + 1
        max_accuracy = 0.0
        if loss_scaler and "grad_scale_manager" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["grad_scale_manager"])
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
    else:        
        logger.info(f"==============> Resuming form {file_name}....................")
        checkpoint = torch.load(file_name, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(msg)
        max_accuracy = 0.0
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            logger.info(f"=> loaded successfully '{file_name}' (epoch {checkpoint['epoch']})")
            if 'max_accuracy' in checkpoint:
                max_accuracy = checkpoint['max_accuracy']

        del checkpoint
        torch.cuda.empty_cache()
    return max_accuracy, start_epoch


def save_checkpoint(ckptdir, config, epoch, model, model_ema, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):

    if config.model.params.deepspeed != '':
        if dist.get_rank() == 0:
            os.makedirs(os.path.join(ckptdir, f'ckpt_epoch_{epoch}'), exist_ok=True) 
            checkpoint_path = os.path.join(ckptdir, f'ckpt_epoch_{epoch}', f'state.pth')
            to_save = {
                'epoch': epoch,
                'config': config,
                'max_accuracy': max_accuracy,
                'model_ema': model_ema.state_dict(),
            }
            if loss_scaler is not None:
                to_save["grad_scale_manager"] = loss_scaler.state_dict()
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            torch.save(to_save, checkpoint_path)
        model.save_checkpoint(save_dir=ckptdir, tag=f'ckpt_epoch_{epoch}')
        print(f"rank[{dist.get_rank()}]: {ckptdir}/{f'ckpt_epoch_{epoch}'} saved")
        dist.barrier()
    else:
        if dist.get_rank() == 0:
            save_state = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'max_accuracy': max_accuracy,
                        'scaler': loss_scaler.state_dict(),
                        'epoch': epoch,
                        'config': config}

            save_path = os.path.join(ckptdir, f'ckpt_epoch_{epoch}.pth')
            logger.info(f"{save_path} saving......")
            torch.save(save_state, save_path)
            logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(config, output_dir):
    if config.model.params.deepspeed != '':
        dirs = os.listdir(output_dir)
        dirs = [d for d in dirs if d.startswith('ckpt_epoch')]
        print(f"All checkpoints founded in {output_dir}: {dirs}")
        if len(dirs) > 0:
            dirs = max([int(d.split('_')[-1]) for d in dirs])
            latest_checkpoint = os.path.join(output_dir, 'ckpt_epoch_{}'.format(dirs))
            print(f"The latest checkpoint founded: {latest_checkpoint}")
            resume_file = latest_checkpoint
        else:
            resume_file = None
    else:
        checkpoints = os.listdir(output_dir)
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
        print(f"All checkpoints founded in {output_dir}: {checkpoints}")
        if len(checkpoints) > 0:
            latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
            print(f"The latest checkpoint founded: {latest_checkpoint}")
            resume_file = latest_checkpoint
        else:
            resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
        
def print_model_submodules_info(model):
    print(f"{'Submodule':<20} | {'Type':<15} | {'Total Parameters (M)':>20} | {'Trainable Parameters (M)':>25}")
    print("-"*90)
    
    total_params = 0
    total_trainable_params = 0
    
    for name, submodule in model.named_children():
        num_params = sum(p.numel() for p in submodule.parameters())
        num_trainable_params = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
        total_params += num_params
        total_trainable_params += num_trainable_params
        
        print(f"{name:<20} | {type(submodule).__name__:<15} | {num_params/1e6:>20.2f} | {num_trainable_params/1e6:>25.2f}")
    
    print("-"*90)
    print(f"{'Total':<20} | {'':<15} | {total_params/1e6:>20.2f}M | {total_trainable_params/1e6:>25.2f}M")
    
def save_model_training_info(savedir, model):
    rec_txt1 = open(os.path.join(savedir, 'rec_para_freeze.txt'), 'w')
    rec_txt2 = open(os.path.join(savedir, 'rec_para_train.txt'), 'w')
    for name, para in model.named_parameters():
        if para.requires_grad is False:
            rec_txt1.write(f'{name}\n')
        else:
            rec_txt2.write(f'{name}\n')
    rec_txt1.close()
    rec_txt2.close()