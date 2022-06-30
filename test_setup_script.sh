# bin/bash
pip freeze | xargs pip uninstall -y
pip install -e .
pip install pytest
pytest