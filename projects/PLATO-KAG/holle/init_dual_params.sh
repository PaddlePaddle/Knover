#/bin/bash
set -eux

KS_PARAM_PATH=$1
RG_PARAM_PATH=$2
DST_PARAM_PATH=$3

rm -rf $DST_PARAM_PATH
mkdir -p $DST_PARAM_PATH

python -u \
    ./projects/PLATO-KAG/tools/init_dual_params.py \
    --ks_param_folder ${KS_PARAM_PATH} \
    --rg_param_folder ${RG_PARAM_PATH} \
    --dst_param_folder ${DST_PARAM_PATH}

rm -rf $KS_PARAM_PATH
rm -rf $RG_PARAM_PATH

python -u \
    ./knover/tools/expand_embedding.py \
    --param_path ${DST_PARAM_PATH} \
    --save_path ${DST_PARAM_PATH} \
    --embedding_name dual_encoder_sent_embedding \
    --embedding_new_size 3

python -u \
    ./knover/tools/expand_embedding.py \
    --param_path ${DST_PARAM_PATH} \
    --save_path ${DST_PARAM_PATH} \
    --embedding_name sent_embedding \
    --embedding_new_size 3

python -u \
    ./knover/tools/expand_embedding.py \
    --param_path ${DST_PARAM_PATH} \
    --save_path ${DST_PARAM_PATH} \
    --embedding_name dual_encoder_pos_embedding \
    --embedding_new_size 512

python -u \
    ./knover/tools/expand_embedding.py \
    --param_path ${DST_PARAM_PATH} \
    --save_path ${DST_PARAM_PATH} \
    --embedding_name pos_embedding \
    --embedding_new_size 512

python -u \
    ./knover/tools/expand_embedding.py \
    --param_path ${DST_PARAM_PATH} \
    --save_path ${DST_PARAM_PATH} \
    --embedding_name dual_encoder_role_embedding \
    --embedding_new_size 32

python -u \
    ./knover/tools/expand_embedding.py \
    --param_path ${DST_PARAM_PATH} \
    --save_path ${DST_PARAM_PATH} \
    --embedding_name role_embedding \
    --embedding_new_size 32

python -u \
    ./knover/tools/expand_embedding.py \
    --param_path ${DST_PARAM_PATH} \
    --save_path ${DST_PARAM_PATH} \
    --embedding_name dual_encoder_word_embedding \
    --embedding_new_size 8004

python -u \
    ./knover/tools/expand_embedding.py \
    --param_path ${DST_PARAM_PATH} \
    --save_path ${DST_PARAM_PATH} \
    --embedding_name word_embedding \
    --embedding_new_size 8004

python -u \
    ./knover/tools/expand_embedding.py \
    --param_path ${DST_PARAM_PATH} \
    --save_path ${DST_PARAM_PATH} \
    --embedding_name mask_lm_out_fc.b_0 \
    --embedding_new_size 8004