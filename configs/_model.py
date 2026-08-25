model = dict(
    type="DenseLocalizer",
    # DFC-only per-frame classifier; num_classes is auto-detected from the dataset
    # class_map by tools/train.py (overrides this placeholder).
    num_classes=3,
    projection=dict(
        type="SGPPyramidProj",
        in_channels=2048,
        out_channels=512,
        sgp_mlp_dim=768,
        arch=(2, 2, 5),  # layers in embed / stem / branch
        downsample_type="max",
        sgp_win_size=[1, 1, 1, 1, 1, 1],
        k=5,
        init_conv_vars=0,
        conv_cfg=dict(kernel_size=3),
        norm_cfg=dict(type="LN"),
        path_pdrop=0.1,
        use_abs_pe=True,
        max_seq_len=768,
        input_noise=0.0,
    ),
    neck=dict(
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
)
