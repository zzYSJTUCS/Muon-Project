import os
import sys
import numpy as np
import torch
import inspect
import json
import copy
import argparse
import random
import wandb
from optim.muon import Muon
import config
from models.utils import get_model
from data.utils import get_dataset
from optim.base import train_base
import distributed


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)


def main(args): 
        
    if wandb.run is not None:
        wandb.finish()  

    if args.wandb:
        wandb.init(
            project=args.wandb_project,  
            name=args.exp_name,  
            config=args,  
            reinit=True  
        )



    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = torch.device(args.device)
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(args.device)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Loading dataset '{args.dataset}'")
    
    data = get_dataset(args) # data is a dict: {'train': train_tokenized, 'val': eval_tokenized}
    if args.data_in_ram:
        data = {'train': np.array(data['train']), 'val': np.array(data['val'])}
        
    print(f"Num training tokens: {len(data['train'])}")
    print(f"Num validation tokens: {len(data['val'])}")
    
    model = get_model(args).to(args.device) # todo: take care of initializing the model if args.use_pretrained != 'none'

    model = distributed_backend.transform_model(model)
    
    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = distributed_backend.translate_model_parameter_name_for_node(p_name)
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt/1e6,))
    if args.opt == 'adamw':
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        opt = torch.optim.AdamW(group_specs, lr=args.lr, betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay, **extra_args)
    elif args.opt == 'Muon':
        # init the optimizer(s)
        hidden_matrix_params = [
            p for n, p in model.transformer.h.named_parameters() if p.ndim >= 2 and "embed" not in n
        ]

        # embed_params: Parameters related to embedding layers (wte and wpe)
        embed_params = [
            p for n, p in model.named_parameters() if "embed" in n
        ]

        # scalar_params: Parameters that are scalars (usually biases or layer normalization)
        scalar_params = [
            p for p in model.parameters() if p.ndim < 2
        ]

        # head_params: Parameters related to the final LM head (lm_head)
        head_params = [model.lm_head.weight]
        if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
            head_params.append(model.lm_head.bias)
        # Define learning rates for different parameter groups
        adam_params = [
            dict(params=head_params, lr=0.008),
            dict(params=embed_params, lr=0.08),
            dict(params=scalar_params, lr=0.04)
        ]

        # AdamW optimizer with parameter groups
        optimizer1 = torch.optim.AdamW(adam_params, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay, eps=1e-10, fused=True)

        # Muon optimizer for hidden matrix parameters
        optimizer2 = Muon(hidden_matrix_params, lr=args.lr, momentum=0.95)

        # Combine both optimizers into a list
        opt = [optimizer1, optimizer2]

    else:
        opt = torch.optim.SGD(group_specs, lr=1e-4, momentum=0.9, weight_decay=args.weight_decay)
    
    if args.scheduler != 'none':
        if args.scheduler in ['cos', 'linear']:
            if args.opt == 'Muon':
                adamw, muon = opt
                scheduler1 = torch.optim.lr_scheduler.OneCycleLR(optimizer=adamw, max_lr=args.lr, total_steps=args.iterations, 
                                                                pct_start=args.warmup_percent, anneal_strategy=args.scheduler, 
                                                                cycle_momentum=False, div_factor=1e2, final_div_factor=.1)
                scheduler2 = torch.optim.lr_scheduler.OneCycleLR(optimizer=muon, max_lr=0.05, total_steps=args.iterations, 
                                                                pct_start=args.warmup_percent, anneal_strategy=args.scheduler, 
                                                                cycle_momentum=False, div_factor=1e2, final_div_factor=.1)
                scheduler = [scheduler1, scheduler2]
            else:    
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, max_lr=args.lr, total_steps=args.iterations, 
                                                                pct_start=args.warmup_percent, anneal_strategy=args.scheduler, 
                                                                cycle_momentum=False, div_factor=1e2, final_div_factor=.1)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    args.world_size = distributed_backend.get_world_size()
    exp_name = args.exp_name
    if distributed_backend.is_master_process() and args.wandb:
        params_copy = copy.deepcopy(vars(args))
        del params_copy['device']
        wandb.init(project=args.wandb_project, name=exp_name, config=params_copy)
    
    ckpt_path = os.path.join(args.results_base_folder, args.dataset, args.model, exp_name)
    if not os.path.exists(ckpt_path):
        if distributed_backend.is_master_process():
            os.makedirs(ckpt_path)
        distributed_backend.sync()
    elif os.path.isfile(os.path.join(ckpt_path, "summary.json")): # the experiment was already completed
        print(f"Already found experiment '{ckpt_path}'.\nSkipping.")
        sys.exit(0)

    itr = 0
    rng_state_dict = None
    if args.use_pretrained == "auto":
        checkpoints = [file for file in os.listdir(ckpt_path) if 'ckpt_' in file]
        if checkpoints:
            args.use_pretrained = sorted(checkpoints)[-1]
        else:
            args.use_pretrained = None
    
    if args.use_pretrained is not None:
        last_ckpt_path = args.use_pretrained
        print(f"Resuming from {last_ckpt_path}")
        checkpoint = torch.load(os.path.join(ckpt_path, last_ckpt_path))
        model_state_dict = {distributed_backend.translate_model_parameter_name_for_node(k.replace("_orig_mod.", ""))[0]:v for k,v in checkpoint['model'].items()}
        # FIXME checkpoints from compiled model have _orig_mod keyword

        optimizer_state_dict = checkpoint['optimizer']
        rng_state_dict = {
            module: checkpoint[module] for module in [
                "cpu_rng_state", 
                "gpu_rng_state", 
                "numpy_rng_state", 
                "py_rng_state",
                "train_sampler_state"
            ]
        }

        model.load_state_dict(model_state_dict) 
        opt.load_state_dict(optimizer_state_dict)
        itr = checkpoint['itr']
        if scheduler is not None:
            scheduler_state_dict = checkpoint['scheduler']
            scheduler.load_state_dict(scheduler_state_dict)

    if args.model in ['base', 'llama2']: # all train functions have the same interface
        train = train_base
    else:
        raise NotImplementedError(f"No training method implemented for model type '{args.model}'.")

    print(f"\nTraining model={args.model} \n{vars(args)}\n")

    stats = train(model, opt, data, args.data_seed, scheduler, args.iterations, args.acc_steps, args.batch_size, args.sequence_length, 
                  eval_freq=args.eval_freq, 
                  distributed_backend=distributed_backend,
                  ckpt_path=f"{ckpt_path}/ckpt.pt", itr=itr, rng_state_dict=rng_state_dict, extra_args=args)
    
    args.device = None
    args.dtype = None
    stats['args'] = vars(args)
    if distributed_backend.is_master_process():
        with open(f"{ckpt_path}/summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


#if __name__ == "__main__":
#   args = get_args()
#    main(args)

if __name__ == "__main__":
    base_args = get_args()  
    
    
    experiments = [ 
        {"lr": 1e-2, "opt": "Muon", "batch_size": 128, "acc_steps": 10, "iterations": 2500, "eval_freq": 20},
        {"lr": 3e-2, "opt": "Muon", "batch_size": 128, "acc_steps": 10, "iterations": 2500, "eval_freq": 20},
        {"lr": 5e-2, "opt": "adamw", "batch_size": 128, "acc_steps": 10, "iterations": 2500, "eval_freq": 20},
    ]


    for exp in experiments:
        args = copy.deepcopy(base_args)
        for key, value in exp.items():
            setattr(args, key, value)
            args.exp_name = f"{args.model}_opt{args.opt}_lr{args.lr}_bs{args.batch_size}x{args.acc_steps}_seqlen{args.sequence_length}_seed={args.seed}"
        main(args)  
