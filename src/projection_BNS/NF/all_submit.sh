#!/bin/bash

# Choose an EOS, and then a range of IDs for which the training script has to be submitted
OUTDIR="./outdir"
TEMPLATE_FILE="template.sh"
EOS="HQC18"

for ID in $(seq 7 9); do
  NEW_SCRIPT="./slurm_scripts/submit_${EOS}_${ID}.sh"

  echo
  echo
  echo

  echo "==== Submitting NF training job for ${EOS} and GW event ID ${ID} ===="
  echo
  
  # Create a unique SLURM script
  cp "$TEMPLATE_FILE" "$NEW_SCRIPT"
  
  # Replace placeholders in the SLURM script
  sed -i "s|{{{EOS}}}|$EOS|g" "$NEW_SCRIPT"
  sed -i "s|{{{ID}}}|$ID|g" "$NEW_SCRIPT"
  
  # Submit the job to SLURM
  sbatch "$NEW_SCRIPT"
  echo "==== Submitted NF training job for ${EOS} and GW event ID ${ID} ===="
  
  echo
  echo
  echo
done