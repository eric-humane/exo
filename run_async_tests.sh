#!/bin/bash
# Script to run async improvement tests

# Activate the virtual environment
source .venv/bin/activate

# Run the async tests
python -m exo.tests.test_async_improvements

# Exit with the status of the test run
exit $?