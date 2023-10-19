python vlr/train.py exp_dir = /home/duytran/Desktop/vlr/outputs/ \
                exp_name = vlr_v1 \
                streaming = True \
                ckpt_path = \
                data.modality = video \
                data.batch_size = 10 \
                data.num_workers = 16 \
                data.max_frames = 750 \
                data.max_frames_val = 500 \
                data.dataset.root_dir = /home/duytran/Downloads/vlr/ \
                data.dataset.train_dir = cropping \
                data.dataset.test_size = 0.1 \
                optimizer.lr = 1e-4 \
                optimizer.weight_decay = 1e-4 \
                optimizer.warmup_epochs = 5 \
                optimizer.name = adamw \
                trainer.max_epochs = 75 \
                trainer.precision = 16-mixed \
                trainer.max_steps = 1000 \
                trainer.accumulate_grad_batches = 1 \
                trainer.log_every_n_steps = 50 \
                trainer.gradient_clip_val = 5.0 \
               
