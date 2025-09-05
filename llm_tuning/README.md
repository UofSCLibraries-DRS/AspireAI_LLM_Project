# Setup

>[!NOTE]
> This file does not cover anything regarding connecting and setting up RCI.
> For information on how to connect and set up RCI, refer the [manaul](https://docs.google.com/document/d/1S4kpOkPnQeoAcIlQKFjZHeql1IsC4dw_oFTRXOGuGLI/edit?usp=sharing)

## Loading Models

Each model that is trained should be manually loaded outside of the script that will run on HPC. Downloading models on compute nodes wastes resources and bogs down the queue.

First, download the Hugging Face Hub CLI.

```bash
pip install -U huggingface_hub[cli]
```

If you need to download gated models (e.g. LLaMA) that require accepting a license agreement, log in with your access token:

```bash
hf auth login
```

Finally, when downloading a model, be sure to download it into the `/work/<USERNAME>/...` directory. Installing a model into `/home/<USERNAME>/...` will fill up your alloted 25GB.

```bash
hf download <MODEL_STRING> --local-dir /work/<USERNAME>/...
```

## Running and Monitoring Scripts

To run your script, navigate to the directory containing the script and run:
```bash
sbatch <NAME OF SCRIPT>.sh
```

This command will give you a job id. I recommend copying this into a file, or noting it down.

To monitor the job's position in the queue, run:

```bash
squeue --job <JOB ID>
```

To monitor all of your jobs run:

```bash
squeue --user <USERNAME>
```

Finally, if you want to cancel a job, use the following:

```bash
scancel <JOB ID>
```

For more information, on commands to manage jobs, refer to the [SLURM docs](https://slurm.schedmd.com/quickstart.html#commands).