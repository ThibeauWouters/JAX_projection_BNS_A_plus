#!/bin/bash

# List of GW event IDs (replace with actual IDs as needed)
OUTDIR="./outdir"
TEMPLATE_FILE="template.sh" # Path to the submission bash template, located in PWD
TEMPLATE_CONFIG=$OUTDIR/template.ini  # Path to the config template, located in $OUTDIR
TEMPLATE_PRIOR=$OUTDIR/template.prior  # Path to the config template, located in $OUTDIR

# Loop over each GW event ID
for ID in $(seq 1 1); do
  EVENT_DIR="${OUTDIR}/injection_${ID}"
  NEW_SCRIPT="./slurm_scripts/submit_${ID}.sh"

  echo
  echo
  echo

  echo "==== Submitting job for GW Event ID ${EVENT_DIR} ===="
  echo
  
  # Check if the event directory exists
  if [ -d "$EVENT_DIR" ]; then
    echo "Directory $EVENT_DIR already exists."
  else
    # Create the directory and copy the template config
    echo "Directory $EVENT_DIR does not exist. Creating it now."
    mkdir -p "$EVENT_DIR"
    echo "Copying template config file to $EVENT_DIR."
    cp "$TEMPLATE_CONFIG" "$EVENT_DIR/config.ini"
    cp "$TEMPLATE_PRIOR" "$EVENT_DIR/prior.prior"
  fi
  
  # Create a unique SLURM script for each GW event
  cp "$TEMPLATE_FILE" "$NEW_SCRIPT"
  
  # Replace placeholders in the SLURM script
  sed -i "s|{{{OUTDIR}}}|$EVENT_DIR|g" "$NEW_SCRIPT"
  
  # Submit the job to SLURM
  sbatch "$NEW_SCRIPT"
  echo "==== Submitted job for GW Event ID ${EVENT_DIR} ===="
  
  echo
  echo
  echo
done