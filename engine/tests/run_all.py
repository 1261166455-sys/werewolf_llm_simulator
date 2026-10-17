'''
Run the whole engine test suite.

    ./venv/bin/python tests/run_all.py            # everything
    ./venv/bin/python tests/run_all.py rules      # only test_rules.py
    ./venv/bin/python tests/run_all.py -k sheriff # only tests matching "sheriff"
'''
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main(argv: list[str]) -> int:
    pattern, keyword = 'test_*.py', None
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == '-k':
            index += 1
            keyword = argv[index] if index < len(argv) else None
        elif argument.startswith('-k'):
            keyword = argument[2:].lstrip('=')
        elif not argument.startswith('-'):
            pattern = 'test_*.py' if argument == 'all' else f'test_{argument}.py'
        index += 1

    suite = unittest.defaultTestLoader.discover(str(HERE), pattern=pattern)

    if keyword:
        def flatten(item):
            if isinstance(item, unittest.TestSuite):
                for child in item:
                    yield from flatten(child)
            else:
                yield item
        suite = unittest.TestSuite(
            test for test in flatten(suite) if keyword.lower() in test.id().lower())

    # the engine prints its own log lines; keep them out of the test output
    # (unittest reports on stderr, so the summary is still visible)
    with contextlib.redirect_stdout(io.StringIO()):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
