# Neural Combinatorial Optimization with RL (Bello et al., 2017) - TSP, PyTorch 2.x
# Reference: github.com/pemami4911/neural-combinatorial-rl-pytorch (MIT)
import math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

def tour_length(coords, tour):
    p = coords.gather(1, tour.unsqueeze(-1).expand(-1, -1, 2))
    return (p - p.roll(-1, dims=1)).norm(dim=-1).sum(1)

def make_dataset(size, n, seed, device="cuda"):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(size, n, 2, generator=g).to(device)

def _vec(d):
    return nn.Parameter(torch.empty(d).uniform_(-1 / math.sqrt(d), 1 / math.sqrt(d)))

def init_uniform(m):
    for p in m.parameters():
        nn.init.uniform_(p, -0.08, 0.08)

class PointerNet(nn.Module):
    def __init__(self, d=128, n_glimpses=1, clip_C=10.0):
        super().__init__()
        self.embed = nn.Linear(2, d)
        self.encoder = nn.LSTM(d, d, batch_first=True)
        self.decoder = nn.LSTMCell(d, d)
        self.g0 = nn.Parameter(torch.empty(d).uniform_(-0.08, 0.08))                  # <g> start token
        self.Wg_ref, self.Wg_q, self.vg = nn.Linear(d, d), nn.Linear(d, d), _vec(d)   # glimpse
        self.Wp_ref, self.Wp_q, self.vp = nn.Linear(d, d), nn.Linear(d, d), _vec(d)   # pointer
        self.n_glimpses, self.C = n_glimpses, clip_C

    def forward(self, coords, greedy=False, T=1.0, keep_probs=False):
        B, n, _ = coords.shape
        ar = torch.arange(B, device=coords.device)
        emb = self.embed(coords)
        enc, (h, c) = self.encoder(emb)
        h, c = h[0], c[0]
        g_ref, p_ref = self.Wg_ref(enc), self.Wp_ref(enc)
        mask = torch.zeros(B, n, dtype=torch.bool, device=coords.device)
        x = self.g0.unsqueeze(0).expand(B, -1)
        tour, logp_sum, probs = [], 0.0, []
        for _ in range(n):
            h, c = self.decoder(x, (h, c))
            q = h
            for _ in range(self.n_glimpses):
                u = (torch.tanh(g_ref + self.Wg_q(q).unsqueeze(1)) @ self.vg).masked_fill(mask, float("-inf"))
                q = torch.einsum("bn,bnd->bd", F.softmax(u, -1), enc)
            u = torch.tanh(p_ref + self.Wp_q(q).unsqueeze(1)) @ self.vp
            if self.C:
                u = self.C * torch.tanh(u)                                   # logit clipping (A.2)
            logp = F.log_softmax(u.masked_fill(mask, float("-inf")) / T, dim=-1)
            idx = logp.argmax(-1) if greedy else torch.multinomial(logp.exp(), 1).squeeze(1)
            tour.append(idx)
            logp_sum = logp_sum + logp[ar, idx]
            if keep_probs:
                probs.append(logp.exp().detach())
            mask = mask | F.one_hot(idx, n).bool()
            x = emb[ar, idx]
        return torch.stack(tour, 1), logp_sum, (torch.stack(probs, 1) if keep_probs else None)

class Critic(nn.Module):
    def __init__(self, d=128, n_process=3):
        super().__init__()
        self.embed = nn.Linear(2, d)
        self.encoder = nn.LSTM(d, d, batch_first=True)
        self.W_ref, self.W_q, self.v = nn.Linear(d, d), nn.Linear(d, d), _vec(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.n_process = n_process

    def forward(self, coords):
        enc, (h, _) = self.encoder(self.embed(coords))
        q, ref = h[0], self.W_ref(enc)
        for _ in range(self.n_process):
            a = F.softmax(torch.tanh(ref + self.W_q(q).unsqueeze(1)) @ self.v, -1)
            q = torch.einsum("bn,bnd->bd", a, enc)
        return self.head(q).squeeze(-1)

@torch.no_grad()
def evaluate(actor, data):
    actor.eval()
    tour, _, _ = actor(data, greedy=True)
    return tour_length(data, tour).mean().item()

@torch.no_grad()
def sample_best(actor, data, K=1280, T=1.0, chunk=131072):
    actor.eval()
    G, best = max(1, chunk // K), []
    for i in range(0, data.shape[0], G):
        x = data[i:i + G].repeat_interleave(K, 0)
        tour, _, _ = actor(x, T=T)
        best.append(tour_length(x, tour).view(-1, K).min(1).values)
    return torch.cat(best).mean().item()

def train(n=20, steps=20000, batch=512, lr=1e-3, baseline="critic", beta=0.8,
          seed=0, device="cuda", eval_every=500, ckpt=None):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    val = make_dataset(1000, n, seed=1234, device=device)
    actor = PointerNet().to(device); init_uniform(actor)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, 5000, 0.96)
    if baseline == "critic":
        critic = Critic().to(device); init_uniform(critic)
        opt_c = torch.optim.Adam(critic.parameters(), lr=lr)
        sch_c = torch.optim.lr_scheduler.StepLR(opt_c, 5000, 0.96)
    ema, hist, t0 = None, [], time.time()
    for step in range(1, steps + 1):
        actor.train()
        coords = torch.rand(batch, n, 2, device=device)
        tour, logp, _ = actor(coords)
        L = tour_length(coords, tour)
        if baseline == "critic":
            b = critic(coords)
            loss_c = F.mse_loss(b, L)
            opt_c.zero_grad(); loss_c.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0); opt_c.step(); sch_c.step()
            b = b.detach()
        else:
            ema = L.mean() if ema is None else beta * ema + (1 - beta) * L.mean()
            b = ema
        loss = ((L - b) * logp).mean()                                       # REINFORCE, eq. (5)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0); opt.step(); sch.step()
        if step == 1 or step % eval_every == 0:
            v = evaluate(actor, val)
            hist.append(dict(step=step, train_len=L.mean().item(), val_greedy=v,
                             minutes=(time.time() - t0) / 60))
            print(f"[{baseline}] step {step:6d} | train {L.mean().item():.3f} | val greedy {v:.3f} | {hist[-1]['minutes']:.1f} min")
    if ckpt:
        torch.save({"actor": actor.state_dict(), "hist": hist, "n": n, "baseline": baseline}, ckpt)
    return actor, hist

def load_actor(path, device="cuda"):
    ck = torch.load(path, map_location=device)
    actor = PointerNet().to(device); actor.load_state_dict(ck["actor"]); actor.eval()
    return actor, ck["hist"]

# ---------- classic baselines (numpy) ----------
def dist_matrix(p):
    return np.linalg.norm(p[:, None] - p[None], axis=-1)

def length_np(p, t):
    c = p[np.append(t, t[0])]
    return np.linalg.norm(np.diff(c, axis=0), axis=1).sum()

def nearest_neighbor(p):
    D, n = dist_matrix(p), len(p)
    tour, seen = [0], np.zeros(n, bool); seen[0] = True
    for _ in range(n - 1):
        j = int(np.where(seen, np.inf, D[tour[-1]]).argmin()); tour.append(j); seen[j] = True
    return np.array(tour)

def two_opt(p, tour):
    D, t, n = dist_matrix(p), tour.copy(), len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b, c, d = t[i - 1], t[i], t[j], t[(j + 1) % n]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                    t[i:j + 1] = t[i:j + 1][::-1].copy(); improved = True
    return t

def lkh(p):
    import elkai
    M = np.rint(dist_matrix(p) * 1e6).astype(int).tolist()
    t = elkai.DistanceMatrix(M).solve_tsp()
    return np.array(t[:-1] if len(t) == len(p) + 1 else t)

def run_baselines(data):
    methods = {"Random tour": lambda p: np.random.permutation(len(p)),
               "Nearest neighbour": nearest_neighbor,
               "NN + 2-opt": lambda p: two_opt(p, nearest_neighbor(p)),
               "LKH (near-optimal)": lkh}
    out = {}
    for name, f in methods.items():
        try:
            t0 = time.time()
            out[name] = (float(np.mean([length_np(p, f(p)) for p in data])), time.time() - t0)
            print(f"{name:20s} {out[name][0]:.4f}  ({out[name][1]:.1f}s)")
        except Exception as e:
            print(f"Skipped {name}: {e}")
    return out
