## 1. Create Conda Environment

Use the following command to create a virtual environment:

```bash
conda create -n almi-rl python=3.8
conda activate unitree-rl
```
### 1.1 Install PyTorch
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 1.2 Install Isaac Gym
#### 1.2.1 Download

Download [Isaac Gym](https://developer.nvidia.com/isaac-gym) from Nvidia’s official website.

#### 1.2.2 Install

After extracting the package, navigate to the `isaacgym/python` folder and install it using the following commands:

```bash
cd isaacgym/python
pip install -e .
```

### 1.3 Install rsl_rl
```bash
cd rsl_rl
pip install -e .
```



### 1.4 Install others
```bash
cd ALMI_RL
pip install -e .
pip install -r requirements_rl.txt
```

## 2. Training and Playing

### 2.1 Training first iteration of lower body

We firstly use open-loop control for upper body and then train lower body policy. We provide a motion file `all_wave.pkl` containing some motions from AMASS datasets. 
If you want to use other motions to set curriculum, you can create file refering this and also a surviving file refering `mean_episode_length.csv` -- We load a baseline lower policy and set [num_motions] envs, and each env load a upper body motion. We set the upper bound steps: 1000 and statistic surviving time for each motion.

``` bash
python legged_gym/scripts/train.py \
    --task=h1_2_wb_curriculum \
    --run_name=lower_body_iteration1 \
    --headless \
```

Then you can play:

``` bash
python legged_gym/scripts/play_curriculum.py --task=h1_2_wb_curriculum --load_run=xxx --checkpoint=xxx
```


### 2.2 Training first iteration of upper body
We load lower-1 policy and then training the upper body policy.
You should set the trained lower-body policy at `h1_2_upper_config.py`.

``` bash
python legged_gym/scripts/train.py \
    --task=h1_2_upper \
    --run_name=upper_body_iteration1 \
    --headless \
```

Then you can play:

``` bash
python legged_gym/scripts/play_curriculum_upper.py --task=h1_2_upper --load_run=xxx --checkpoint=xxx
```

### 2.3 Training second iteration of upper body

This training is similar to 2.1, except that using upper-1 policy instead of open-loop control.
You should set the trained upper-body policy at `h1_2_lower_config.py`.

``` bash
python legged_gym/scripts/train.py \
    --task=h1_2_lower \
    --run_name=lower_body_iteration2 \
    --headless \
```

Then you can play:

``` bash
python legged_gym/scripts/play_curriculum_lower.py --task=h1_2_lower --load_run=xxx --checkpoint=xxx
```

### 2.4 Training second iteration of upper body
We load lower-2 policy and then training the upper body policy.
You should set the trained lower-body policy at `h1_2_upper_config.py`.

``` bash
python legged_gym/scripts/train.py \
    --task=h1_2_upper \
    --run_name=upper_body_iteration2 \
    --headless \
```

Then you can play:

``` bash
python legged_gym/scripts/play_curriculum_upper.py --task=h1_2_upper --load_run=xxx --checkpoint=xxx
```