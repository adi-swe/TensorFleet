#!/bin/sh
# Real classical-ML sweeps; requires the documented running Docker test stack.
exec python3 "$(dirname "$0")/research_suite.py" run "$@"
