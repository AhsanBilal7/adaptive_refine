#!/bin/bash
# =============================================================================
# Comprehensive Ablation Study Script
# =============================================================================
# 1. Similarity mode: cosine vs NLI vs hybrid
# 2. NLI model comparison: roberta, deberta, mDeBERTa
# 3. Lambda penalty sweep: 0.0, 0.5, 1.0, 1.5, 2.0
# 4. Hybrid top-k sweep: 3, 5, 10, 15, 20
# 5. NLI summary vs full trajectory
# 6. Clustering: kmeans (K=3,5,10), hierarchical (C=0.80,0.90,0.95), soft
# 7. Temperature sensitivity: Tb and Ts
# =============================================================================
set -e

pip install latex2sympy2 word2number FlagEmbedding "transformers>=4.36.0" scikit-learn scipy datasets pyyaml accelerate

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/allmodels/Qwen2.5-Math-PRM-7B"

export CONFIG=recipes/Llama-3.2-1B-Instruct/dora_nli_qwen.yaml
export prm_path=allmodels/Qwen2.5-Math-PRM-7B
export model_path=allmodels/Llama-3.2-1B-Instruct

export nli_roberta=FacebookAI/roberta-large-mnli
export nli_deberta=allmodels/deberta-large-mnli
export nli_mdeberta=allmodels/mDeBERTa-v3-base-mnli-xnli

N=64

echo "=== ABLATION 1: Similarity Mode Comparison ==="
for mode in cosine nli hybrid; do
    output_dir=result/MATH500/ablation_sim_mode/${mode}
    for seed in 0 1 2 3 4; do
        extra=""
        if [ "$mode" = "nli" ] || [ "$mode" = "hybrid" ]; then
            extra="--nli_model_path=$nli_roberta --nli_lambda_penalty=1.0"
        fi
        [ "$mode" = "hybrid" ] && extra="$extra --nli_top_k=10"
        python scripts/test_time_compute.py $CONFIG \
            --n=$N --balance_alpha=0.01 --seed=$seed \
            --prm_path=$prm_path --model_path=$model_path \
            --similarity_mode=$mode $extra --output_dir=$output_dir
    done
done

echo "=== ABLATION 2: NLI Model Comparison ==="
for nli_model in $nli_roberta $nli_deberta $nli_mdeberta; do
    nli_name=$(basename $nli_model)
    output_dir=result/MATH500/ablation_nli_model/${nli_name}
    for seed in 0 1 2 3 4; do
        python scripts/test_time_compute.py $CONFIG \
            --n=$N --balance_alpha=0.01 --seed=$seed \
            --prm_path=$prm_path --model_path=$model_path \
            --similarity_mode=nli --nli_model_path=$nli_model \
            --nli_lambda_penalty=1.0 --output_dir=$output_dir
    done
done

echo "=== ABLATION 3: Lambda Penalty Sweep ==="
for lam in 0.0 0.5 1.0 1.5 2.0; do
    output_dir=result/MATH500/ablation_lambda/lambda_${lam}
    for seed in 0 1 2 3 4; do
        python scripts/test_time_compute.py $CONFIG \
            --n=$N --balance_alpha=0.01 --seed=$seed \
            --prm_path=$prm_path --model_path=$model_path \
            --similarity_mode=nli --nli_model_path=$nli_roberta \
            --nli_lambda_penalty=$lam --output_dir=$output_dir
    done
done

echo "=== ABLATION 4: Hybrid Top-K Sweep ==="
for topk in 3 5 10 15 20; do
    output_dir=result/MATH500/ablation_topk/topk_${topk}
    for seed in 0 1 2 3 4; do
        python scripts/test_time_compute.py $CONFIG \
            --n=$N --balance_alpha=0.01 --seed=$seed \
            --prm_path=$prm_path --model_path=$model_path \
            --similarity_mode=hybrid --nli_model_path=$nli_roberta \
            --nli_lambda_penalty=1.0 --nli_top_k=$topk \
            --output_dir=$output_dir
    done
done

echo "=== ABLATION 5: NLI Summary vs Full ==="
for use_summary in true false; do
    tag="full"; [ "$use_summary" = "true" ] && tag="summary"
    output_dir=result/MATH500/ablation_summary/${tag}
    for seed in 0 1 2 3 4; do
        python scripts/test_time_compute.py $CONFIG \
            --n=$N --balance_alpha=0.01 --seed=$seed \
            --prm_path=$prm_path --model_path=$model_path \
            --similarity_mode=nli --nli_model_path=$nli_roberta \
            --nli_lambda_penalty=1.0 --nli_use_summary=$use_summary \
            --output_dir=$output_dir
    done
done

echo "=== ABLATION 6: Clustering Methods ==="
for k in 3 5 10; do
    output_dir=result/MATH500/ablation_cluster/kmeans_k${k}
    for seed in 0 1 2 3 4; do
        python scripts/test_time_compute.py $CONFIG \
            --n=$N --balance_alpha=0.01 --seed=$seed \
            --prm_path=$prm_path --model_path=$model_path \
            --similarity_mode=cosine --ablation_similarity=kmeans \
            --ablation_kmeans_k=$k --output_dir=$output_dir
    done
done
for c in 0.80 0.90 0.95; do
    output_dir=result/MATH500/ablation_cluster/hier_c${c}
    for seed in 0 1 2 3 4; do
        python scripts/test_time_compute.py $CONFIG \
            --n=$N --balance_alpha=0.01 --seed=$seed \
            --prm_path=$prm_path --model_path=$model_path \
            --similarity_mode=cosine --ablation_similarity=hierarchical \
            --ablation_hierarchical_threshold=$c --output_dir=$output_dir
    done
done
# Soft (default DORA)
output_dir=result/MATH500/ablation_cluster/soft_default
for seed in 0 1 2 3 4; do
    python scripts/test_time_compute.py $CONFIG \
        --n=$N --balance_alpha=0.01 --seed=$seed \
        --prm_path=$prm_path --model_path=$model_path \
        --similarity_mode=cosine --output_dir=$output_dir
done

echo "=== ABLATION 7: Temperature Sensitivity ==="
for tb in 0.01 0.1 1.0; do
    output_dir=result/MATH500/ablation_temp/nli_tb_${tb}
    for seed in 0 1 2 3 4; do
        python scripts/test_time_compute.py $CONFIG \
            --n=$N --balance_alpha=0.01 --rebase_temperature=$tb \
            --seed=$seed --prm_path=$prm_path --model_path=$model_path \
            --similarity_mode=nli --nli_model_path=$nli_roberta \
            --nli_lambda_penalty=1.0 --output_dir=$output_dir
    done
done
for ts in 0.001 0.01 0.1 1.0; do
    output_dir=result/MATH500/ablation_temp/nli_ts_${ts}
    for seed in 0 1 2 3 4; do
        python scripts/test_time_compute.py $CONFIG \
            --n=$N --balance_alpha=$ts --seed=$seed \
            --prm_path=$prm_path --model_path=$model_path \
            --similarity_mode=nli --nli_model_path=$nli_roberta \
            --nli_lambda_penalty=1.0 --output_dir=$output_dir
    done
done

echo "All ablations complete!"
