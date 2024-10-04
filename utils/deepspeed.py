import os
import torch
import torch.distributed as dist
import json


def create_ds_config(args, config, cfgdir):
    config.deepspeed_config = os.path.join(cfgdir, f"deepspeed_config_{dist.get_rank()}.json")
    opt_lower = config.trainer.optimizer.lower()
    assert opt_lower == 'adamw', "deepspeed only support adamw"
    with open(config.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": config.data.params.batch_size * config.trainer.accumulate_grad_batches * dist.get_world_size(),
            "train_micro_batch_size_per_gpu": config.data.params.batch_size,
            "steps_per_print": 10,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": config.model.base_learning_rate,
                    "weight_decay": config.model.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9, 0.999
                    ],
                    "eps": 1e-8
                }
            },
        }
        
        if 'fp32' in config.model.params.deepspeed:
            ds_config["fp16"] = {
                "enabled": False}
        else:
            ds_config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": config.trainer.initial_scale,
                "loss_scale_window": 128}

        if config.trainer.clip_grad > 0.0:
            ds_config["gradient_clipping"] = config.trainer.clip_grad
        zero_opt = int(config.model.params.deepspeed.split('_')[-1])
        if zero_opt == 1:
            ds_config["zero_optimization"] = {"stage": zero_opt}
        elif zero_opt == 2:
            ds_config["zero_optimization"] = {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu",
                    },
                    "contiguous_gradients": True,
                    "overlap_comm": True
                }
        writer.write(json.dumps(ds_config, indent=2))
