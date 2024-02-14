### Future Plan
- [ ] Implement wandb monitoring
- [ ] Add logic for resuming training
- [ ] Combine logging and tqdm module, see best practices

### Current Todo

- annotation related
    - [ ] (docstring) model/layers.py
    - [ ] (docstring) model/model.py
    - [ ] (docstring) model/losses.py
    - [ ] (docstring) model/metrics.py
- logging related
    - [ ] Review `tb_writer_fns` in logger.py
    - [ ] Review `n` in `MetricTracker.update()`
    - [ ] Save the configuration as a file in each training
- training related
    - [ ] Review on how to handle `run_id`: "", no id, with id
    - [ ] Review model saving with the latest best checkpoint, not every `save_period` (considering `early_step`)

### Done ✓

- [x] Create README.md
- [x] config.py: add `run_id` to cli logic
- [x] Move any logger function to logger.py
- [x] data_loader.py: Implement custom transformation for the dataset
- [x] trainer.py: Remove iteration-based training logic 
- [x] (docstring) utils/ module
- [x] (docstring) train.py
