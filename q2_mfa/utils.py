# ----------------------------------------------------------------------------
# Copyright (c) 2026, QIIME 2 development team..
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

EXTERNAL_CMD_WARNING = (
    "Running external command line application(s). "
    "This may print messages to stdout and/or stderr.\n"
    "The command(s) being run are below. These commands "
    "cannot be manually re-run as they will depend on "
    "temporary files that no longer exist."
)


def run_command(cmd, cwd, verbose=True, env=None):
    """
    Runs an external command-line application.

    Prints the command when verbose mode is enabled and delegates execution to
    subprocess.run with check=True.

    Args:
        cmd (list[str]):
            Command and arguments to run.
        cwd (str | None):
            Working directory for the command.
        verbose (bool):
            Whether to print the command before running it.
        env (dict | None):
            Optional environment variables for the subprocess.
    """
    if verbose:
        print(EXTERNAL_CMD_WARNING)
        print("\nCommand:", end=" ")
        print(" ".join(cmd), end="\n\n")
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def run_r_script(script_name, params, package_name):
    """
    Constructs and runs a command-line call to an R script.

    Parameters are passed as command-line flags in the form `--name value`.
    Values set to None are skipped.

    Args:
        script_name (str):
            Base name of the installed R script without the `.R` extension.
        params (dict):
            Parameter names and values to pass as command-line arguments.
        package_name (str):
            Name of the invoked package for error messages.

    Raises:
        Exception:
            If the R script returns a non-zero exit status.
    """
    cmd = [f"{script_name}.R"]

    for key, value in params.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])

    try:
        run_command(cmd, verbose=True, cwd=None)
    except subprocess.CalledProcessError as error:
        raise Exception(
            f"An error was encountered while running {package_name}, "
            f"(return code {error.returncode}), please inspect "
            "stdout and stderr to learn more."
        )


def run_r_table_script(
    table: pd.DataFrame,
    script_name: str,
    parameters: dict[str, object],
    package_name: str,
) -> pd.DataFrame:
    """
    Runs an R script that reads and writes table data through temporary TSVs.

    Creates temporary input and output table paths, adds them to the provided
    parameter dictionary, runs the installed R script, and reads the output TSV
    back into a DataFrame with the original index and columns restored.

    Args:
        table (pd.DataFrame):
            Input table to pass to the R script.
        script_name (str):
            Base name of the installed R script without the `.R` extension.
        parameters (dict[str, object]):
            Script-specific parameters to pass as named command-line flags.
        package_name (str):
            Name of the invoked package for error messages.

    Returns:
        pd.DataFrame:
            Output table read from the R script output path.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "input.tsv"
        output_path = temp_dir / "output.tsv"

        table.to_csv(input_path, sep="\t", na_rep="NA")

        params = {
            "input_path": input_path,
            "output_path": output_path,
            **parameters,
        }
        run_r_script(
            script_name=script_name,
            params=params,
            package_name=package_name,
        )

        output = pd.read_csv(output_path, sep="\t", index_col=0)

    output.index = table.index
    output.columns = table.columns
    return output
