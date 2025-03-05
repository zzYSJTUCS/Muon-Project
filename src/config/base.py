import distributed


def none_or_str(value):
    if value == "None":
        return None
    return value


def parse_args(base_parser, args, namespace):
    parser = base_parser

    # General training params
    parser.add_argument("--run_prefix", default=None, type=str)
    parser.add_argument("--experiment_name", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--data_seed", default=1337, type=int)
    parser.add_argument("--eval_interval", default=200, type=int)
    parser.add_argument("--full_eval_at", nargs="+", type=int)
    parser.add_argument("--eval_batches", default=32, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument(
        "--distributed_backend",
        default=None,
        type=str,
        required=False,
        choices=distributed.registered_backends(),
    )
    parser.add_argument("--log_interval", default=50, type=int)

    # Checkpointing
    parser.add_argument("--results_base_folder", default="./exps", type=str)
    parser.add_argument("--permanent_ckpt_interval", default=0, type=int)
    parser.add_argument("--latest_ckpt_interval", default=0, type=int)
    parser.add_argument("--resume_from", default=None, type=str)
    parser.add_argument("--resume_from_swa", default=None, type=str)

    parser.add_argument("--auto_resume", default=True)

    # logging params (WandB)
    parser.add_argument("--wandb", action="store_true")  # whether to use wandb or not
    parser.add_argument("--wandb_project", default="my-project", type=str)
    parser.add_argument(
        "--wandb_run_prefix", default="none", type=str
    )  # is added before the autogenerated experiment name
    parser.add_argument(
        "--eval_seq_prefix", default="none", type=str
    )  # prefix used to generate sequences
    parser.add_argument("--log_dynamics", action="store_true")
    parser.add_argument(
        "--dynamics_logger_cfg", default="./src/logger/rotational_logger.yaml", type=str
    )
    parser.add_argument("--wandb_entity", default=None, type=none_or_str)
    parser.add_argument("--log_parameter_norms", action="store_true")
    parser.add_argument("--norm_order", default=2)

    # Schedule
    parser.add_argument(
        "--scheduler",
        default="cos",
        choices=["linear", "cos", "wsd", "none", "cos_inf", "cos_wsd", "dd"],
    )
    parser.add_argument(
        "--final_div_factor", default=1, type=float
    )  # cosine and linear schedulers
    parser.add_argument("--cos_inf_steps", default=0, type=int)
    # parser.add_argument("--cos-final-lr", default=1e-6, type=float)
    parser.add_argument("--iterations", default=15000, type=int)
    parser.add_argument("--warmup_steps", default=3000, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    # wsd
    parser.add_argument("--wsd_final_lr_scale", default=0.0, type=float)
    parser.add_argument("--wsd_fract_decay", default=0.1, type=float)
    # parser.add_argument("--wsd-exponential-decay", action="store_true")
    parser.add_argument(
        "--decay_type",
        default="linear",
        choices=["linear", "cosine", "exp", "miror_cosine", "square", "sqrt"],
    )
    parser.add_argument(
        "--dd_second_decay_type",
        default="linear",
        choices=["linear", "cosine", "exp", "miror_cosine", "square", "sqrt"],
    )
    parser.add_argument("--dd_first_lr_factor", default=1e-2, type=float)

    # Optimization
    parser.add_argument(
        "--opt",
        default="adamw",
        choices=[
            "adamw",
            "sgd",
            "muon",
            "soap",
            "ademamix",
            "adoptademamix",
            "lion",
            "sf-adamw",
            "sf-sgd",
            "adam-mini",
            "signsgd",
            "signum",
            "sgdf",
            "prodigy",
            "sophiag",
            "shampoo",
            "adopt",
            "clip-adagrad",
            "clip-adagrad-delay-eta",
            "clip-adam",
            "clip-adam-delay-eta",
            "mars",
            "adafactor",
            "lamb",
            "normalized-sgd",
            "sgd-with-adam",
            "scion",
            "scion-light",
        ],
    )
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--acc_steps", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument(
        "--grad_clip", default=1.0, type=float
    )  # default value is 1.0 in NanoGPT
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--shampoo_beta", default=-1.0, type=float)
    parser.add_argument("--precondition_frequency", default=10, type=int)
    parser.add_argument("--max_precond_dim", default=10000, type=int)
    parser.add_argument("--merge_dims", default=False, type=bool)
    parser.add_argument("--precondition_1d", default=False, type=bool)
    parser.add_argument("--normalize_grads", default=False, type=bool)
    parser.add_argument("--soap_data_format", default="channels_first", type=str)
    parser.add_argument("--correct_bias", default=True, type=bool)
    parser.add_argument("--nesterov", default=False, type=bool)
    parser.add_argument("--muon_ns_steps", default=5, type=int)
    parser.add_argument("--muon_lr_factor", default=1.0, type=float)
    parser.add_argument("--adema_beta3", default=0.9, type=float)
    parser.add_argument("--adema_alpha", default=2.0, type=float)
    parser.add_argument("--adema_beta3_warmup", default=None, type=int)
    parser.add_argument("--adema_alpha_warmup", default=None, type=int)
    parser.add_argument("--schedulefree_r", default=0.0, type=float)
    parser.add_argument("--weight_lr_power", default=2.0, type=float)
    parser.add_argument("--model_sharding", default=None, type=bool)
    parser.add_argument("--adam_mini_verbose", default=False, type=bool)
    parser.add_argument("--dampening", default=0.0, type=float)
    parser.add_argument("--prodigy_beta3", default=None, type=float)
    parser.add_argument("--prodigy_decouple", default=True, type=bool)
    parser.add_argument("--prodigy_use_bias_correction", default=False, type=bool)
    parser.add_argument("--prodigy_safeguard_warmup", default=False, type=bool)
    parser.add_argument("--prodigy_fsdp_in_use", default=False, type=bool)
    parser.add_argument("--sophia_rho", default=0.04, type=float)
    parser.add_argument("--sophia_bs", default=480, type=int)
    parser.add_argument(
        "--clipping_type", default="no", choices=["no", "local", "elementwise"]
    )
    parser.add_argument("--clip_eta", default=1.0, type=float)
    parser.add_argument(
        "--mars_type",
        default="mars-adamw",
        choices=["mars-adamw", "mars-lion", "mars-shampoo"],
    )
    parser.add_argument("--mars_vr_gamma", default=0.025, type=float)
    parser.add_argument("--mars_is_approx", default=True, type=float)
    parser.add_argument("--mars_lr", default=3e-3, type=float)
    parser.add_argument("--mars_beta1", default=0.95, type=float)
    parser.add_argument("--mars_beta2", default=0.99, type=float)
    parser.add_argument("--adafactor_decay_rate", default=-0.8, type=float)
    parser.add_argument("--lamb_use_bias_correction", default=False, type=bool)
    parser.add_argument("--proj_norms", default=False, action="store_true")
    parser.add_argument("--proj_embeds", default=False, action="store_true")
    parser.add_argument("--proj_logits", default=False, action="store_true")
    parser.add_argument("--sgd_sign_update", default=False, action="store_true")
    parser.add_argument("--sign_norm", default=False, action="store_true")
    parser.add_argument("--normalized", default=False, action="store_true")
    parser.add_argument("--sgd_lr_scale", default=1.0, type=float)
    parser.add_argument("--adopt_decouple", default=True, type=bool)
    parser.add_argument("--adopt_eps", default=1e-6, type=float)
    parser.add_argument("--cautious", default=False, type=bool)
    parser.add_argument(
        "--weight_decay_scheduler",
        default=None,
        choices=["linear", "cos", "stable-decay", "wsd"],
    )
    parser.add_argument("--final_weight_decay", default=0.1, type=float)

    # Dataset params
    parser.add_argument("--datasets_dir", type=str, default="./src/data/datasets/")
    parser.add_argument(
        "--dataset",
        default="slimpajama",
        choices=[
            "wikitext",
            "shakespeare-char",
            "arxiv",
            "arxiv2000",
            "arxiv+wiki",
            "openwebtext2",
            "redpajama",
            "slimpajama",
            "slimpajama_chunk1",
            "redpajamav2",
            "fineweb",
            "finewebedu",
            "c4",
        ],
    )
    parser.add_argument(
        "--tokenizer", default="gpt2", type=str, choices=["gpt2", "mistral"]
    )
    parser.add_argument("--vocab_size", default=50304, type=int)
    parser.add_argument(
        "--data_in_ram", action="store_true"
    )  # force the data to RAM, mostly useless except for openwebtext2

    # Model params
    parser.add_argument(
        "--model",
        default="llama",
        choices=[
            "base",
            "llama",
            "test",
        ],
    )
    parser.add_argument("--parallel_block", action="store_true")
    parser.add_argument(
        "--use_pretrained", default="none", type=str
    )  # 'none', 'gpt-2' or a path to the pretraind model
    parser.add_argument("--from_dense", action="store_true")
    parser.add_argument("--init_std", default=0.02, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--n_head", default=12, type=int)
    parser.add_argument("--n_layer", default=24, type=int)  # depths in att + ff blocks
    parser.add_argument("--sequence_length", default=512, type=int)
    parser.add_argument(
        "--n_embd", default=768, type=int  # embedding size / hidden size ...
    )
    parser.add_argument(
        "--multiple_of",  # make SwiGLU hidden layer size multiple of large power of 2
        default=256,
        type=int,
    )
    parser.add_argument("--n_kv_head", default=None, type=int)  # for Adam-mini
    parser.add_argument("--rmsnorm_eps", default=1e-5, type=float)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        type=str,
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--bias", default=False, type=bool)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--mlp_dim_exp_factor", default=1.0, type=float)

    return parser.parse_args(args, namespace)
