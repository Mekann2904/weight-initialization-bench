# /Users/mekann/github/weight-initialization-bench/mnist_init_compare.py
# MNISTの初期化比較と可視化をまとめたスクリプト。
# 初期化の違いによる学習挙動と対称性を数値と図で確認するために存在する。
# 関連ファイル: data/, README.md

import os
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# =========================
# 設定
# =========================
SEED = 0
DEVICE = "cpu"  # 対称性(完全一致)を見たいなら cpu 推奨。gpuだと極小の差が出ることがある。
BATCH_SIZE = 128
FIG_DIR = "figures"

# 学習比較（ReLU 5層）を軽くしたいならサブセットを使う
TRAIN_SUBSET = 10000   # None なら全訓練(60000)
TEST_SUBSET  = 2000    # None なら全テスト(10000)

EPOCHS_COMPARE = 8     # std=0.01 / Xavier / He の比較エポック数
LR_COMPARE = 0.01

# 対称性デモ（Sigmoid 1層）: 少なめで十分
EPOCHS_SYMM = 5
LR_SYMM = 0.1
H_SYMM = 100

# アクティベーション分布（Sigmoid 5層）
HIST_BATCH = 1000
HIST_HIDDEN = 100
HIST_LAYERS = 5


# =========================
# 再現性
# =========================
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # さらに厳密にしたい場合
set_seed(SEED)


# =========================
# 図の保存ユーティリティ
# =========================
def save_fig(fig, name: str):
    """図を指定パスに保存して閉じる。"""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"[FIG] saved -> {path}")
    plt.close(fig)


# =========================
# MNIST 読み込み（正規化して0付近中心に寄せる）
# =========================
def load_mnist_loaders(batch_size=128, train_subset=None, test_subset=None):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    if train_subset is not None:
        train_ds = Subset(train_ds, list(range(train_subset)))
    if test_subset is not None:
        test_ds = Subset(test_ds, list(range(test_subset)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# =========================
# 対称性指標
# =========================
def max_row_diff(W: torch.Tensor) -> float:
    """
    W: (num_units, in_features) のように「ユニットごとに行」が対応する重み。
    行同士がどれだけ違うかの最大値（0なら完全一致＝対称性が崩れていない）。
    """
    W = W.detach().cpu()
    base = W[0:1, :]
    return float((W - base).abs().max().item())

def max_col_diff(W: torch.Tensor) -> float:
    """
    W: (out_features, num_units) のように「ユニットごとに列」が対応する重み。
    列同士がどれだけ違うかの最大値。
    """
    W = W.detach().cpu()
    base = W[:, 0:1]
    return float((W - base).abs().max().item())

def hidden_unit_variance(h: torch.Tensor) -> float:
    """
    h: (batch, hidden_units)
    サンプルごとの「ユニット間分散」を平均。0なら全ユニットが同じ出力。
    """
    h = h.detach()
    return float(torch.var(h, dim=1, unbiased=False).mean().item())


# =========================
# 1) 「全ゼロ初期化で対称性が崩れない」デモ（Sigmoid 1層）
# =========================
class OneHiddenSigmoid(nn.Module):
    def __init__(self, hidden=100):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.sigmoid(self.fc1(x))
        y = self.fc2(h)
        return y, h

def init_all_zero(model: nn.Module):
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

def init_small_normal(model: nn.Module, std=0.01):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.normal_(mean=0.0, std=std)
                m.bias.zero_()

def train_symmetry_demo(train_loader, test_loader):
    models = {
        "all_zero": OneHiddenSigmoid(H_SYMM).to(DEVICE),
        "rand0.01": OneHiddenSigmoid(H_SYMM).to(DEVICE),
    }
    init_all_zero(models["all_zero"])
    init_small_normal(models["rand0.01"], std=0.01)

    history = {k: {"train_acc": [], "test_acc": [], "sym_fc1": [], "sym_fc2": [], "h_var": []}
               for k in models.keys()}

    for name, model in models.items():
        opt = torch.optim.SGD(model.parameters(), lr=LR_SYMM)

        for ep in range(EPOCHS_SYMM):
            model.train()
            correct = total = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits, h = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()

                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

            train_acc = correct / total

            # 対称性：fc1は「行」がユニット、fc2は「列」がユニット
            sym_fc1 = max_row_diff(model.fc1.weight)         # 0に張り付くなら隠れ層が同一
            sym_fc2 = max_col_diff(model.fc2.weight)         # 0に張り付くなら出力側も同一
            hvar = hidden_unit_variance(h)                   # 0に張り付くなら隠れ層出力が同一

            test_acc = evaluate(model, test_loader)

            history[name]["train_acc"].append(train_acc)
            history[name]["test_acc"].append(test_acc)
            history[name]["sym_fc1"].append(sym_fc1)
            history[name]["sym_fc2"].append(sym_fc2)
            history[name]["h_var"].append(hvar)

            print(f"[SYMM][{name}] ep={ep} train_acc={train_acc:.3f} test_acc={test_acc:.3f} "
                  f"sym(fc1,row)={sym_fc1:.3e} sym(fc2,col)={sym_fc2:.3e} hidden_var={hvar:.3e}")

    # プロット
    epochs = np.arange(EPOCHS_SYMM)

    fig, ax = plt.subplots()
    for name in models.keys():
        ax.plot(epochs, history[name]["test_acc"], label=f"{name} test_acc")
    ax.set_title("Symmetry demo (Sigmoid 1-hidden): test accuracy")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend(); ax.grid(True)
    save_fig(fig, "symmetry_test_accuracy")

    fig, ax = plt.subplots()
    for name in models.keys():
        ax.plot(epochs, history[name]["sym_fc1"], label=f"{name} sym_fc1(row)")
    ax.set_title("Symmetry demo: fc1 symmetry (0 means all hidden units identical)")
    ax.set_xlabel("epoch"); ax.set_ylabel("max |W_i - W_0|"); ax.set_yscale("log")
    ax.legend(); ax.grid(True)
    save_fig(fig, "symmetry_fc1")

    fig, ax = plt.subplots()
    for name in models.keys():
        ax.plot(epochs, history[name]["h_var"], label=f"{name} hidden_var")
    ax.set_title("Symmetry demo: hidden activation variance (0 means identical outputs)")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean var across hidden units"); ax.set_yscale("log")
    ax.legend(); ax.grid(True)
    save_fig(fig, "symmetry_hidden_variance")


# =========================
# 評価
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits, _ = model(xb) if isinstance(model, OneHiddenSigmoid) else (model(xb), None)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total


# =========================
# 2) アクティベーション分布（Sigmoid 5層）をMNISTで観察
#    std=1 / std=0.01 / Xavier(1/sqrt(n)) を比較
# =========================
def sigmoid_activation_histograms(train_loader):
    xb, _ = next(iter(train_loader))
    xb = xb[:HIST_BATCH].to(DEVICE)
    x = xb.view(xb.size(0), -1)

    def forward_sigmoid_5layers(x, init_mode):
        activations = []
        in_dim = 784
        dim = HIST_HIDDEN
        h = x

        for layer_idx in range(HIST_LAYERS):
            fan_in = in_dim if layer_idx == 0 else dim

            if init_mode == "std=1":
                std = 1.0
            elif init_mode == "std=0.01":
                std = 0.01
            elif init_mode == "xavier(1/sqrt(n))":
                std = 1.0 / math.sqrt(fan_in)
            else:
                raise ValueError(init_mode)

            W = torch.randn(fan_in, dim, device=DEVICE) * std
            z = h @ W
            h = torch.sigmoid(z)
            activations.append(h.detach().cpu().numpy())

            in_dim = dim

        return activations

    modes = ["std=1", "std=0.01", "xavier(1/sqrt(n))"]
    acts_by_mode = {m: forward_sigmoid_5layers(x, m) for m in modes}

    # 3行 x 5列のヒストグラム
    fig, axes = plt.subplots(len(modes), HIST_LAYERS, figsize=(16, 7))
    for r, mode in enumerate(modes):
        for c in range(HIST_LAYERS):
            a = acts_by_mode[mode][c].flatten()
            ax = axes[r, c]
            ax.hist(a, bins=30, range=(0, 1))
            if r == 0:
                ax.set_title(f"{c+1}-layer")
            if c == 0:
                ax.set_ylabel(mode)
    fig.suptitle("Activation histograms (Sigmoid, MNIST batch)")
    fig.tight_layout()
    save_fig(fig, "sigmoid_activation_histograms")


# =========================
# 3) MNIST学習で初期値比較（ReLU 5層, 各100）
#    std=0.01 / Xavier / He を比較してプロット
# =========================
class MLPReLU(nn.Module):
    def __init__(self, hidden_sizes, input_dim=784, num_classes=10):
        super().__init__()
        dims = [input_dim] + hidden_sizes + [num_classes]
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs) - 1:
                x = F.relu(x)
        return x

def init_mlp(model: MLPReLU, mode: str):
    with torch.no_grad():
        for fc in model.fcs:
            fan_in = fc.weight.size(1)  # (out,in)
            if mode == "std=0.01":
                std = 0.01
            elif mode == "xavier(1/sqrt(n))":
                std = 1.0 / math.sqrt(fan_in)
            elif mode == "he(sqrt(2/n))":
                std = math.sqrt(2.0 / fan_in)
            else:
                raise ValueError(mode)

            fc.weight.normal_(0.0, std)
            fc.bias.zero_()

def train_compare_inits(train_loader, test_loader):
    modes = ["std=0.01", "xavier(1/sqrt(n))", "he(sqrt(2/n))"]
    hidden = [100] * 5

    results = {m: {"train_loss": [], "test_acc": []} for m in modes}

    for mode in modes:
        model = MLPReLU(hidden).to(DEVICE)
        init_mlp(model, mode)

        opt = torch.optim.SGD(model.parameters(), lr=LR_COMPARE)

        for ep in range(EPOCHS_COMPARE):
            model.train()
            total_loss = 0.0
            n = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()

                total_loss += loss.item() * yb.size(0)
                n += yb.size(0)

            train_loss = total_loss / n
            test_acc = eval_mlp(model, test_loader)

            results[mode]["train_loss"].append(train_loss)
            results[mode]["test_acc"].append(test_acc)

            print(f"[COMPARE][{mode}] ep={ep} train_loss={train_loss:.4f} test_acc={test_acc:.3f}")

    # プロット：loss
    epochs = np.arange(EPOCHS_COMPARE)
    fig, ax = plt.subplots()
    for mode in modes:
        ax.plot(epochs, results[mode]["train_loss"], label=mode)
    ax.set_title("MNIST (ReLU 5-hidden x100): train loss vs epoch")
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss"); ax.legend(); ax.grid(True)
    save_fig(fig, "compare_train_loss")

    # プロット：test acc
    fig, ax = plt.subplots()
    for mode in modes:
        ax.plot(epochs, results[mode]["test_acc"], label=mode)
    ax.set_title("MNIST (ReLU 5-hidden x100): test accuracy vs epoch")
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy"); ax.legend(); ax.grid(True)
    save_fig(fig, "compare_test_accuracy")


@torch.no_grad()
def eval_mlp(model, loader):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total


# =========================
# 参考：ReLUで全ゼロ初期化は「完全に止まる」ことの確認（補助）
# =========================
def relu_allzero_stuck_check(train_loader):
    model = MLPReLU([100]*5).to(DEVICE)
    with torch.no_grad():
        for fc in model.fcs:
            fc.weight.zero_()
            fc.bias.zero_()

    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)

    for step in range(3):
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()

        # 勾配がゼロに張り付く（更新できない）ことを確認
        gnorm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                gnorm += float(p.grad.detach().abs().sum().item())

        opt.step()
        print(f"[ReLU all-zero stuck] step={step} loss={loss.item():.4f} grad_L1sum={gnorm:.3e}")


# =========================
# main
# =========================
def main():
    train_loader, test_loader = load_mnist_loaders(
        batch_size=BATCH_SIZE,
        train_subset=TRAIN_SUBSET,
        test_subset=TEST_SUBSET
    )

    # 1) 対称性が崩れないことを「数値とプロット」で確認（Sigmoid 1層）
    train_symmetry_demo(train_loader, test_loader)

    # 2) アクティベーション分布（Sigmoid 5層）をMNISTで観察
    sigmoid_activation_histograms(train_loader)

    # 3) MNISTで初期値比較（ReLU 5層）：std=0.01 / Xavier / He
    train_compare_inits(train_loader, test_loader)

    # 補助：ReLUで全ゼロ初期化が完全に止まること
    relu_allzero_stuck_check(train_loader)


if __name__ == "__main__":
    main()
