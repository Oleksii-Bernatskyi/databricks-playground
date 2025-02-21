import pytest
import os
import sys
import subprocess
import pyspark

import importlib
import tests
importlib.reload(tests)


def run_test():
    # Skip writing pyc files on a readonly filesystem.
    sys.dont_write_bytecode = True

    # Run pytest.
    retcode = pytest.main(["tests.py", "-v", "--tb=short", "-p", "no:cacheprovider"])

    # Fail the cell execution if there are any test failures.
    # assert retcode == 0, "The pytest invocation failed. See the log for details."
    if retcode == 0:
        print("All tests passed! ✅")
    else:
        print("Some tests failed. ❌")
        exit(retcode)  # Return the exit code to signal test failure


if __name__ == "__main__":
    run_test()

