# Example Snakemake rule. config["resources"] is your <resources> folder.
#
# <resources>/SLANT must look like the image's /extra:
#   model_dir/<piece>/finetune_out/models/model_epoch_0027.pth      (27 pieces)
#   full-multi-atlas/MNI/average305_t1_tal_lin.nii.gz, ..._mask.nii.gz
#   full-multi-atlas/atlas-processing/atlas_model.npz               (or .mat)
#   full-multi-atlas/niftyreg/bin, full-multi-atlas/ANTs-bin/bin    (optional, see env file)
# "slantpy extract pipeline/slant:v1 <resources>/SLANT" creates exactly this.

SLANT = config["resources"] + "/SLANT"

rule slant:
    input:
        t1="{sample}/t1.nii.gz",
        weights=SLANT + "/model_dir/2_2_2/finetune_out/models/model_epoch_0027.pth",
    output:
        seg="{sample}/slant/FinalResult/t1_seg.nii.gz",
        vol="{sample}/slant/FinalVolTxt/t1_label_volumes.txt",
    params:
        out="{sample}/slant",
        extra=SLANT,
    threads: 4
    resources:
        mem_mb=6000,      # estimate with headroom; the network needs about 2 GB
    conda:
        "envs/slantpy.yaml"
    shell:
        "slantpy run {input.t1} {params.out} --extra-dir {params.extra} "
        "--threads {threads} --no-pdf"
