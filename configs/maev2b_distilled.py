"""`--model maev2b-distilled` — `maev2b` started from distilled adapters.

Identical network to `configs/maev2b.py`, key for key. The only difference is
where the adapters begin: instead of random init on top of the K400 base, the
whole backbone is seeded from `calms21_vitB_distilled_best.pth`, whose 108
adapter tensors were FD-distilled from a V-JEPA 2 ViT-L teacher (see
`tools/distill_jepa_to_vmae.py`).

That checkpoint is a full trained model, so its projection/neck/head tensors do
not belong to a backbone and are dropped on load; the backbone loader keeps the
269 keys that do. Downstream heads are rebuilt for the new class map either way.
"""
_base_ = ["maev2b.py"]

model = dict(
    backbone=dict(
        custom=dict(pretrain="pretrained/calms21_vitB_distilled_best.pth"),
    ),
)

work_dir = "runs/maev2b_distilled"
