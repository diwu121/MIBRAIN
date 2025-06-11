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
)

train_setting = dict(
    output_dir='/root/siton-data-wudiPersonal/data/AiT/outputs/',
    label_smoothing = 0.1,
    seed=42,
    sub_count=1,
    eval_epoch=1,
    use_cls_loss=False,
    use_recon_loss = True,
    cls_cost = 1.0,
    recon_cost = 0.0,
    cls_count = 23,
    data=dict(
        data_folder="/root/siton-data-wudiPersonal/DATA/EEG_SPEECH/all_sub/std_data/initial/",
    ),
    opt_params=dict(
        epochs=10,
        batch_size=32,
        learning_rate=1e-5,
        min_lr=1e-6,
        warmup_ratio=1e-3,
        warmup_steps=500,
        weight_decay=0.005,
        schedule_type='cosine',
    )
)

test_setting = dict(
    data=dict(
    data_folder="/root/siton-data-wudiPersonal/DATA/EEG_SPEECH/all_sub/std_data/initial/",
    ),
)
