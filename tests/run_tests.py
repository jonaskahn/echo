#!/usr/bin/env python3
"""Test runner script for Echo framework tests."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(test_path=None, markers=None, verbose=False, coverage=True):
    """Run pytest with specified options."""
    cmd = ["python", "-m", "pytest"]

    if test_path:
        cmd.append(test_path)

    if markers:
        cmd.extend(["-m", markers])

    if verbose:
        cmd.append("-v")

    if not coverage:
        cmd.append("--no-cov")

    print(f"Running tests with command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code: {e.returncode}")
        return e.returncode


def main():
    """Main function for test runner."""
    parser = argparse.ArgumentParser(description="Run Echo framework tests")
    parser.add_argument("test_path", nargs="?", help="Specific test file or directory to run")
    parser.add_argument("-m", "--markers", help="Run only tests matching given markers (e.g., 'unit', 'integration')")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--list-tests", action="store_true", help="List all available tests without running them")

    args = parser.parse_args()

    if args.list_tests:
        cmd = ["python", "-m", "pytest", "--collect-only", "-q"]
        if args.test_path:
            cmd.append(args.test_path)
        subprocess.run(cmd)
        return 0

    return run_tests(
        test_path=args.test_path, markers=args.markers, verbose=args.verbose, coverage=not args.no_coverage
    )


if __name__ == "__main__":
    sys.exit(main())
