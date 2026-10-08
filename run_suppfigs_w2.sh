#!/bin/bash

#SBATCH --job-name=make_suppfigs_w2
#SBATCH --output=%x.o%j
#SBATCH --partition=mem
#SBATCH --cpus-per-task=3
#SBATCH --mem=500G

# Supplementary figures that use the inverse-Wasserstein (W2) ensemble
# weighting: ED Figs. 4, 5, 6, 8, 9, 11, 12, 13.

ROOT=/gpfs/workdir/shared/juicce/RE_Colin/code_review/WSED-WSBD
SUPP=$ROOT/supp_figs

# Load necessary modules
module purge
module load anaconda3/2023.09-0/none-none
export OMP_NUM_THREADS=1

# Optional: reduce ESMF noise and internal logging
export ESMF_RUNTIME_PROFILE=OFF
export ESMF_RUNTIME_TRACE=OFF

# A failing script is reported but does not stop the following ones.
run() {
    echo "=== $(date '+%F %T') $1 ==="
    python -u "$SUPP/$1" || echo "!!! $1 failed (exit $?)"
}

# --- SWED figures (share fig3.py, which uses the xclim env's ESMF) ---
source activate /gpfs/workdir/shared/juicce/envs/xclim
run suppfig4_valuebyalpha_all_gwl.py
run suppfig5_projected_change_wasserstein.py
conda deactivate

# --- SWBD / duration figures (share fig45.py, xenv) ---
source activate /gpfs/workdir/shared/juicce/envs/xenv
run suppfig6_duration_distribution_swed_swbd.py
run suppfig8_driver_effects.py
run suppfig9_re_share_effect.py
run suppfig11_mix_gwl_effects.py
run suppfig12_combined_threshold_sensitivity.py
run suppfig13_demand_sensitivity.py
