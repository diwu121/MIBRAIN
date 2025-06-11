from torchvision import transforms as T

num_token = 256
commitment_cost = 0.25
seq_len = 1536
is_public_only= True
is_pretrained_code = False
model = dict(
    model_type='MaskedModeling',
    num_resnet_blocks=2,
    downsample_ratio=4,
    num_tokens=16,
    codebook_dim=16,
    region_dim=32,
    channel_per_region=31,
    brain_region_count=21,
    hidden_dim=128,
    use_norm=False,
    train_objective='regression',
    max_value=1.,
    residul_type='v1',
    loss_type='mse',
    drop_rate=0.1,
    drop_path_rate=0.1,
    num_heads=8,
    init_values=0.1,
    ffn_channels_ratio=2,
    use_pretrain_proto=True,
)

train_setting = dict(
    output_dir='/root/siton-data-wudiPersonal/data/AiT/outputs/',
    pretrain_proto_path="/root/data/AiT/outputs/20240702_084535_pretrain_H2DiLR_dim128_l4_h4/clustering_codebook.pth",
    label_smoothing = 0.1,
    seed=42,
    sub_count=1,
    eval_epoch=1,
    use_cls_loss=True,
    use_recon_loss = True,
    cls_cost = 1.0,
    recon_cost = 0.0,
    cls_count = 23,
    data=dict(
        # data_folder="/root/siton-data-wudiPersonal/DATA/sEEG-Speech/standardized_data_padded_initial/std_padded_data_init_512_sublabel/",
        data_folder="/root/siton-data-wudiPersonal/DATA/EEG_SPEECH/wc/std_data/initial/",
    ),
    opt_params=dict(
        epochs=200,
        batch_size=32,
        learning_rate=1e-4,
        min_lr=1e-5,
        warmup_ratio=1e-3,
        warmup_steps=500,
        weight_decay=0.005,
        schedule_type='cosine',
    )
)

test_setting = dict(
    data=dict(
    # data_folder="/root/siton-data-wudiPersonal/DATA/sEEG-Speech/standardized_data_padded_initial/std_padded_data_init_512_sublabel/",
    data_folder="/root/siton-data-wudiPersonal/DATA/EEG_SPEECH/wc/std_data/initial/",
    ),
)
