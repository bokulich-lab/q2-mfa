# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Laboratories.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import subprocess
import unittest
from unittest.mock import call, patch

import pandas as pd
import pandas.testing as pdt
from rachis.plugin.testing import TestPluginBase

from q2_mfa.utils import (
    EXTERNAL_CMD_WARNING,
    run_command,
    run_r_script,
    run_r_table_script,
)


class TestRunCommand(TestPluginBase):
    package = "q2_mfa.tests"

    @patch("subprocess.run")
    @patch("builtins.print")
    def test_run_command_verbose(self, mock_print, mock_subprocess_run):
        # Mock command and working directory
        cmd = ["echo", "Hello"]
        cwd = "/test/directory"

        # Run the function with verbose=True
        run_command(cmd, cwd=cwd, verbose=True)

        # Check if subprocess.run was called with the correct arguments
        mock_subprocess_run.assert_called_once_with(cmd, check=True, cwd=cwd, env=None)

        # Check if the correct print statements were called
        mock_print.assert_has_calls(
            [
                call(EXTERNAL_CMD_WARNING),
                call("\nCommand:", end=" "),
                call("echo Hello", end="\n\n"),
            ]
        )

    @patch("subprocess.run")
    @patch("builtins.print")
    def test_run_command_non_verbose(self, mock_print, mock_subprocess_run):
        # Mock command and working directory
        cmd = ["echo", "Hello"]
        cwd = "/test/directory"

        # Run the function with verbose=False
        run_command(cmd, cwd=cwd, verbose=False)

        # Check if subprocess.run was called with the correct arguments
        mock_subprocess_run.assert_called_once_with(cmd, check=True, cwd=cwd, env=None)

        # Ensure no print statements were made
        mock_print.assert_not_called()

    @patch("subprocess.run")
    def test_run_r_script_success(self, mock_subprocess):
        # Call function
        run_r_script(
            params={"param1": "value1", "param2": 42, "param3": None},
            script_name="test_script",
            package_name="q2_ms",
        )

        # Check if subprocess.run was called correctly
        expected_cmd = [
            "test_script.R",
            "--param1",
            "value1",
            "--param2",
            "42",
        ]

        mock_subprocess.assert_called_once_with(
            expected_cmd, check=True, cwd=None, env=unittest.mock.ANY
        )

    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd"))
    def test_run_r_script_failure(self, mock_subprocess):
        with self.assertRaises(Exception) as context:
            run_r_script("", {}, "r_package")

        self.assertIn("r_package", str(context.exception))

    @patch("q2_mfa.utils.run_r_script")
    def test_run_r_table_script_round_trips_table_and_parameters(self, mock_run):
        table = pd.DataFrame(
            {"feature-b": [1.0, float("nan")], "feature-a": [3.0, 4.0]},
            index=["sample-2", "sample-1"],
        )

        def write_r_output(script_name, params, package_name):
            table.to_csv(params["output_path"], sep="\t")

        mock_run.side_effect = write_r_output

        observed = run_r_table_script(
            table=table,
            script_name="impute",
            parameters={"method": "knn"},
            package_name="imputeLCMD",
        )

        mock_run.assert_called_once()
        pdt.assert_frame_equal(observed, table)
