python launch.py --predict \
        --config pretrain/fridge/config/parsed.yaml \
        --resume pretrain/fridge/ckpt/last.ckpt \
        dataset.pred_scale=0.1 \
        dataset.view_idx=6 \
        dataset.pred_mode='anim'