# Minimal mrunner experiment configuration
import os

from munch import Munch

from mrunner.experiment import Experiment

# if "NEPTUNE_API_TOKEN" not in os.environ or "NEPTUNE_PROJECT_NAME" not in os.environ:
#     print("Please set NEPTUNE_API_TOKEN and NEPTUNE_PROJECT_NAME env variables")
#     print("Their values can be from up.neptune.ml. Click help and then quickstart.")
#     exit(1)
project_name = "pmtest/llm-random"
import datetime

experiments_list = []
for lr in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
    for use_byte_embeddings in [False, True]:
        exp = Experiment(
            name="Basic experiment",
            script="srun python3 MaxText/train.py",
            project=project_name,
            tags=["test_experiments"],
            env={
                "NEPTUNE_API_TOKEN": os.environ["NEPTUNE_API_TOKEN"],
                "PYTHON_PATH": "$PYTHONPATH:.",
                "SLURM_STEP_NODELIST": "$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')",
                "HF_TOKEN": os.environ["HF_TOKEN"],
            },
            parameters=Munch(
                from_config="MaxText/configs/mpioro/500m_flash.yml",
                run_name=f"runner_pretraining_{datetime.datetime.now()}",
                base_output_directory="gs://focused-llama/mpioro/subtokenizer/500m_lr_grid",
                neptune_project=project_name,
                learning_rate=lr,
                steps=1600,
                eval_interval=0,
                per_device_batch_size=64,
                use_byte_embeddings=use_byte_embeddings
            ),
        )
        experiments_list.append(exp)
