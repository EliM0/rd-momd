# forl-project

Executing the different algorithms:

```bash
python src/evaluation.py --config=<config_dir> --algorithm=['afp', 'dmomd', 'rnn-momd', 'fo-dmomd', 'po-dmomd', 'po-rnn-momd']
```

Check plots:
```bash
tensorboard --logdir=runs
```