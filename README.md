<div align="justify">

# CT-Scan-Lung-Cancer-Classification


- **Logging Documentation**: https://docs.python.org/3/library/logging.html
- **Custom Exception Documentation**: https://docs.python.org/3/tutorial/errors.html

# Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml


# MLFlow
- Documentation: https://mlflow.org/docs/latest/index.html

*CMD*
```bash
mlflow ui
```

# Dagshub
- Documentation: https://dagshub.com/

```python
import dagshub
dagshub.init(repo_owner='MannShrestha', repo_name='CT-Scan-Lung-Cancer-Classification', mlflow=True)
```

# DVC cmd

1. dvc init
2. dvc repro
3. dvc dag
