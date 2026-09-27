# Neural Combinatorial Optimization with RL — TSP (Midterm)

**Saina Zardosht & Shazan Ansar Mohammed**

Reproduction of *Neural Combinatorial Optimization with Reinforcement Learning*
(Bello, Pham, Le, Norouzi, Bengio · Google Brain · [arXiv:1611.09940](https://arxiv.org/abs/1611.09940))
in PyTorch 2, trained on one NVIDIA A100 (Google Colab Pro).

## Question
The paper trains a pointer network with REINFORCE and a **critic network** baseline.
The public reference code reports that a **moving-average** baseline works better.
We trained both with identical settings on TSP20 and compared.

## Results (1,000 fixed TSP20 test graphs, gap vs LKH = 3.845)

| Method | Avg length | Gap |
|---|---|---|
| LKH (near-optimal) | 3.845 | 0.0% |
| RL sampling ×1280, T=2 · critic | 3.908 | 1.6% |
| NN + 2-opt | 3.946 | 2.6% |
| RL greedy · critic | 4.020 | 4.6% |
| RL greedy · moving average | 4.191 | 9.0% |
| Nearest neighbour | 4.496 | 16.9% |

- The critic halves the greedy gap (4.6% vs 9.0%); the moving average was unstable around step 15k.
- Sampling 1,280 tours matches LKH exactly on 29% of graphs and beats 2-opt.
- ~20 minutes of training per model. One seed per method (see limitations).

![Training curves](figs/fig_curve_modern.png)

## Files
- `ncorl.py`: pointer network, critic, REINFORCE training, classic baselines
- `notebooks/541_midterm_experiment.ipynb`: full experiment (Colab)
- `results/`: result table, training histories
- `figs/`: figures used in the slides

## Run
Open the notebook in Colab (GPU runtime) and run the cells in order.

## Limitations and next steps
One seed per method; temperature chosen on the test set; shorter training than the paper; no Active Search yet.
Final project: Active Search, 3 seeds, TSP50/100, train-20 → test-50 generalization, Transformer encoder.

## Credits
Paper and original figures: Bello et al., 2017. Reference implementation:
[pemami4911/neural-combinatorial-rl-pytorch](https://github.com/pemami4911/neural-combinatorial-rl-pytorch) (MIT).
