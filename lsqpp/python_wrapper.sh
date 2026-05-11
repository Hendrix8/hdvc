#!/bin/bash
# Wrapper to run Python with faiss environment
conda run -n dtwrl_env2 python3 "$@"
exit $?
