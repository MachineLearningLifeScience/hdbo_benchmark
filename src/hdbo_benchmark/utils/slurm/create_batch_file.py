import os
import stat
from pathlib import Path

from jinja2 import Template

from hdbo_benchmark.utils.constants import ROOT_DIR, WANDB_PROJECT
from hdbo_benchmark.utils.slurm import __file__ as SLURM_UTILS_PATH


def read_template():
    SLURM_UTILS_DIR = Path(SLURM_UTILS_PATH).parent
    with open(SLURM_UTILS_DIR / "batch_template.sht", "r") as f:
        return f.read()


def write_batch_script_for_commands(
    commands: list[str],
    *,
    job_name: str = "hdbo_benchmark",
    output_dir: str = f"slurm_logs/{WANDB_PROJECT}",
    error_dir: str = f"slurm_logs/{WANDB_PROJECT}",
    gpu_resources: str = "--gres=gpu:titanx:1",
    parallel_count: int = 1,
    slurm_script_output_path: Path | str = ROOT_DIR / "batch_script.local.sh",
    instruction_file_output_path: Path | str = ROOT_DIR / "lines_to_run.local.sh",
):
    template_string = read_template()

    # Create a Jinja2 Template object
    template = Template(template_string)

    # Computing the linecount
    line_count = len(commands)

    # Writing the instruction file
    with open(instruction_file_output_path, "w") as fp_instruction:
        fp_instruction.writelines(commands)

    # Making sure the output and error dirs exist
    output_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    # Define the values for the placeholders
    context = {
        "job_name": job_name,
        "output_path": output_dir,
        "error_path": error_dir,
        "gpu_resources": gpu_resources,
        "line_count": line_count,
        "parallel_count": parallel_count,
        "commands_file": instruction_file_output_path,
    }

    # Render the template with the context
    rendered_script = template.render(context)

    # Save the rendered script to a file
    with open(slurm_script_output_path, "w") as fp_slurm_script:
        fp_slurm_script.write(rendered_script)
        os.chmod(
            slurm_script_output_path,
            os.stat(slurm_script_output_path).st_mode | stat.S_IEXEC,
        )
