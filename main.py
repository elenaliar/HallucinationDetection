import numpy as np
import re
import urllib.request
import os
import time
from collections import Counter


# dataset 

def fetch_text():
    cache = "corpus.txt"
    if os.path.exists(cache):
        with open(cache, "r") as f:
            return f.read()

    # Frankenstein – public domain, reasonable size (~430 k chars)
    url = "https://www.gutenberg.org/files/84/84-0.txt"
    print("downloading corpus …")
    with urllib.request.urlopen(url) as r:
        raw = r.read().decode("utf-8", errors="ignore")

    # strip gutenberg header/footer
    start = raw.find("*** START OF")
    end   = raw.find("*** END OF")
    text  = raw[start:end] if start != -1 else raw

    with open(cache, "w") as f:
        f.write(text)
    return text


def tokenise(text):
    return re.findall(r"[a-z]+", text.lower())


# vocabulary

def build_vocab(tokens, min_count=5):
    freq = Counter(tokens)
    vocab = [w for w, c in freq.items() if c >= min_count]
    vocab.sort()                         # deterministic ordering

    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = vocab
    counts = np.array([freq[w] for w in vocab], dtype=np.float64)
    return w2i, i2w, counts


def subsample_probs(counts, t=1e-4):
    """
    Mikolov's subsampling: frequent words get dropped with probability
    1 - sqrt(t / freq_ratio). Returns keep probability per token.
    """
    freq  = counts / counts.sum()
    probs = 1.0 - np.sqrt(t / np.maximum(freq, 1e-12))
    return np.clip(probs, 0.0, 1.0)


def build_noise_table(counts, power=0.75, table_size=int(1e6)):
    """
    Sample negatives from the smoothed unigram distribution.
    Raising counts to 0.75 gives rare words a slightly better shot.
    """
    smoothed = counts ** power
    smoothed /= smoothed.sum()
    table = np.random.choice(len(counts), size=table_size, p=smoothed)
    return table


# data pipeline 

def make_pairs(token_ids, window):
    """Yield (center, context) pairs with a random window size each step."""
    for i, center in enumerate(token_ids):
        w = np.random.randint(1, window + 1)
        lo = max(0, i - w)
        hi = min(len(token_ids), i + w + 1)
        for j in range(lo, hi):
            if j != i:
                yield center, token_ids[j]


#model parameters 
def init_weights(vocab_size, embed_dim):
    # center embeddings: small uniform init 
    W_in  = (np.random.rand(vocab_size, embed_dim) - 0.5) / embed_dim
    # output / context embeddings: start at zero
    W_out = np.zeros((vocab_size, embed_dim), dtype=np.float64)
    return W_in, W_out


# skip-gram negative sampling 

def sigmoid(x):
    # clip to avoid overflow in exp for very negative values
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def sgns_step(center, context, negatives, W_in, W_out, lr):
    """
    One update for a (center, context) pair.

    Loss = -log σ(v_c · u_o)  -  Σ log σ(-v_c · u_n)

    Returns the scalar loss for logging.
    """
    v_c = W_in[center]           # (D,)
    u_o = W_out[context]         # (D,)
    u_n = W_out[negatives]       # (k, D)

    # forward
    pos_score = np.dot(v_c, u_o)
    neg_scores = u_n @ v_c       # (k,)

    pos_sig = sigmoid(pos_score)
    neg_sig = sigmoid(-neg_scores)

    loss = -np.log(pos_sig + 1e-12) - np.sum(np.log(neg_sig + 1e-12))

    # gradients w.r.t. output vectors
    # ∂L/∂u_o  = (σ(v_c·u_o) - 1) · v_c
    grad_u_o = (pos_sig - 1.0) * v_c

    # ∂L/∂u_n  = (1 - σ(-v_c·u_n)) · v_c  for each negative
    grad_u_n = ((1.0 - neg_sig))[:, None] * v_c[None, :]   # (k, D)

    # gradient flowing back to the center vector
    # accumulate from positive and all negatives before touching W_in
    grad_v_c  = (pos_sig - 1.0) * u_o
    grad_v_c += ((1.0 - neg_sig) @ u_n)   # (D,)

    # SGD updates
    W_in[center]     -= lr * grad_v_c
    W_out[context]   -= lr * grad_u_o
    W_out[negatives] -= lr * grad_u_n

    return loss


# training 

def train(
    W_in, W_out,
    token_ids,
    noise_table,
    drop_prob,       # per-word-id keep probability (for subsampling)
    window=5,
    n_neg=5,
    n_epochs=3,
    lr_start=0.025,
    lr_min=0.0001,
    log_every=100_000,
):
    total_pairs  = 0
    total_tokens = len(token_ids) * n_epochs

    for epoch in range(1, n_epochs + 1):
        # subsampling: randomly skip frequent tokens each epoch
        kept = [t for t in token_ids if np.random.rand() > drop_prob[t]]

        running_loss = 0.0
        pair_count   = 0
        t0           = time.time()

        for step, (center, context) in enumerate(make_pairs(kept, window)):
            total_pairs += 1

            # linearly decay lr based on token progress (Mikolov's schedule)
            progress = min(total_pairs / (total_tokens * 5), 1.0)
            lr = max(lr_start * (1.0 - progress), lr_min)

            # draw negatives, re-sample if we accidentally hit the context word
            negs = noise_table[np.random.randint(0, len(noise_table), n_neg)]
            negs[negs == context] = noise_table[np.random.randint(0, len(noise_table))]

            loss = sgns_step(center, context, negs, W_in, W_out, lr)
            running_loss += loss
            pair_count   += 1

            if pair_count % log_every == 0:
                elapsed = time.time() - t0
                avg_loss = running_loss / log_every
                kpairs_s = log_every / elapsed / 1000
                print(
                    f"  epoch {epoch}  pairs={pair_count:>9,}  "
                    f"loss={avg_loss:.4f}  lr={lr:.5f}  "
                    f"speed={kpairs_s:.1f} kp/s"
                )
                running_loss = 0.0
                t0 = time.time()

        print(f"epoch {epoch} done – {pair_count:,} pairs processed")


# evaluation 

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def nearest_neighbours(word, W_in, w2i, i2w, topk=8):
    if word not in w2i:
        return []
    idx = w2i[word]
    vec = W_in[idx]
    # normalise all vectors once for cosine comparison
    norms  = np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-12
    normed = W_in / norms
    sims   = normed @ (vec / (np.linalg.norm(vec) + 1e-12))
    sims[idx] = -1                   # exclude the query word itself
    top    = np.argsort(sims)[::-1][:topk]
    return [(i2w[i], float(sims[i])) for i in top]


# main 

def main():
    np.random.seed(42)

    # hyper-parameters
    EMBED_DIM = 100
    WINDOW    = 5
    N_NEG     = 5
    N_EPOCHS  = 3
    MIN_COUNT = 10

    print("loading corpus …")
    text   = fetch_text()
    tokens = tokenise(text)
    print(f"  raw tokens : {len(tokens):,}")

    w2i, i2w, counts = build_vocab(tokens, min_count=MIN_COUNT)
    print(f"  vocabulary : {len(i2w):,} words")

    # map corpus to integer ids, drop OOV
    token_ids = [w2i[t] for t in tokens if t in w2i]
    print(f"  kept tokens: {len(token_ids):,}")

    drop_prob   = subsample_probs(counts)
    noise_table = build_noise_table(counts)

    W_in, W_out = init_weights(len(i2w), EMBED_DIM)

    print("\ntraining …")
    train(
        W_in, W_out,
        token_ids,
        noise_table,
        drop_prob,
        window=WINDOW,
        n_neg=N_NEG,
        n_epochs=N_EPOCHS,
    )

    # quick sanity check
    print("\nnearest neighbours:")
    for probe in ["monster", "beautiful", "death", "science", "man"]:
        nbrs = nearest_neighbours(probe, W_in, w2i, i2w)
        if nbrs:
            nbrs_str = ", ".join(f"{w}({s:.2f})" for w, s in nbrs)
            print(f"  {probe:12s} → {nbrs_str}")

    np.save("embeddings_in.npy",  W_in)
    np.save("embeddings_out.npy", W_out)
    print("\nembeddings saved to embeddings_in.npy / embeddings_out.npy")


if __name__ == "__main__":
    main()
