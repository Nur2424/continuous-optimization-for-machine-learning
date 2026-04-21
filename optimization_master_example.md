# Continuous Optimization (§7.1): Engineering-Level Worked Example

One problem. Four data points. Every concept of the chapter — gradient descent, step size, condition number, momentum, SGD, mini-batches, noise — derived on the same example. Everything is numeric. Everything is verifiable.

---

## SECTION 1 — THE PROBLEM WE'RE SOLVING

### The data

We have 4 data points. One feature `x`, one target `y`.

| n | xₙ | yₙ |
|---|----|----|
| 1 | 1  | 2  |
| 2 | 2  | 3  |
| 3 | 3  | 5  |
| 4 | 4  | 6  |

### The model

We fit a line:

```
ŷ = w·x + b
```

- `w` = slope
- `b` = intercept
- `ŷ` = model's prediction

We want to find `(w, b)` that makes `ŷₙ` as close as possible to `yₙ` across all 4 points.

### Why this tiny example is the whole chapter

Linear regression is the **mother of all ML losses**:
- It has a sum-over-data form → SGD applies
- It has a quadratic loss → the Hessian is constant → condition number is exact, not approximate
- It has a closed-form optimum → we can check our numerical answer
- Near any minimum of any smooth loss (even a neural net), the function **looks like** this problem locally (Taylor expansion)

Master this, and §7.1 is yours.

---

## SECTION 2 — THE LOSS (the "sum over data" form)

### Per-point loss

For one data point:

```
Lₙ(w, b) = (w·xₙ + b − yₙ)²
```

It's the squared error for that one point.

### Total loss

The total loss is the **sum** over all points — equation (7.13):

```
L(w, b) = Σ Lₙ(w, b)       (summing n = 1 to 4)
        = Σ (w·xₙ + b − yₙ)²
```

### 🔑 Key observation

Every ML loss has this sum form (or a mean, which is the sum divided by N).

**Why this matters:** because `L` is a sum, its gradient is also a sum. And a sum can be **estimated from a subset**. This single fact is why SGD exists.

---

## SECTION 3 — THE GRADIENT (derive it step by step)

We need the gradient of `L` with respect to `w` and `b`. Chain rule — nothing else.

### Setup the residual

Let `eₙ = w·xₙ + b − yₙ` = the error at point `n`. Then `Lₙ = eₙ²`.

### Partial derivatives (chain rule)

For one point:

```
∂Lₙ/∂w = 2·eₙ · ∂eₙ/∂w = 2·eₙ·xₙ
∂Lₙ/∂b = 2·eₙ · ∂eₙ/∂b = 2·eₙ·1 = 2·eₙ
```

For the total loss, sum over n:

```
∂L/∂w = Σ 2·eₙ·xₙ
∂L/∂b = Σ 2·eₙ
```

### The gradient vector

Pack into a column:

```
∇L = [ ∂L/∂w ]   =   [ 2·Σ eₙ·xₙ ]
     [ ∂L/∂b ]       [ 2·Σ eₙ    ]
```

### 🔑 Key observation

`∇L = Σ ∇Lₙ`. **Gradient of a sum = sum of gradients.** This is why a mini-batch gradient is an unbiased estimator of the full gradient (Section 10).

---

## SECTION 4 — THE INITIAL STATE (step 0, pen-and-paper)

### Starting point

Initialize `w₀ = 0`, `b₀ = 0`. Model predicts `ŷ = 0` for every point.

### Residuals at step 0

Since `ŷ = 0`, we get `eₙ = 0 − yₙ = −yₙ`:

| n | xₙ | yₙ | eₙ  | 2·eₙ·xₙ | 2·eₙ  |
|---|----|----|-----|---------|-------|
| 1 | 1  | 2  | −2  | −4      | −4    |
| 2 | 2  | 3  | −3  | −12     | −6    |
| 3 | 3  | 5  | −5  | −30     | −10   |
| 4 | 4  | 6  | −6  | −48     | −12   |
|   |    |    | **Σ** | **−94** | **−32** |

### Gradient at step 0

```
∂L/∂w = −94
∂L/∂b = −32

∇L₀ = [ −94 ]
      [ −32 ]
```

### Initial loss

```
L₀ = (−2)² + (−3)² + (−5)² + (−6)²
   = 4 + 9 + 25 + 36
   = 74
```

### 🧠 Reading the gradient

Both components are **negative**. That means: L decreases if we **increase** `w` and `b`. We were predicting all zeros but the targets are all positive — yes, we should increase `w` and `b`. The math agrees with common sense.

---

## SECTION 5 — ONE STEP OF GRADIENT DESCENT

### The update rule (equation 7.6)

```
[ w₁ ]   [ w₀ ]         [ ∂L/∂w ]
[ b₁ ] = [ b₀ ]  − γ  ·  [ ∂L/∂b ]
```

Or in words: **new parameters = old parameters − (learning rate) × (gradient)**.

| symbol | meaning | value at step 0 |
|--------|---------|-----------------|
| `γ`    | step size (learning rate) | we pick it |
| `∇L`   | gradient | `[−94, −32]` |
| `(w, b)` | parameters | start at `(0, 0)` |

### Try γ = 0.01 (reasonable)

```
w₁ = 0 − 0.01·(−94) = 0.94
b₁ = 0 − 0.01·(−32) = 0.32
```

New residuals and new loss:

| n | ŷₙ = 0.94·xₙ + 0.32 | eₙ = ŷₙ − yₙ | eₙ² |
|---|---------------------|---------------|------|
| 1 | 1.26                | −0.74         | 0.55 |
| 2 | 2.20                | −0.80         | 0.64 |
| 3 | 3.14                | −0.86         | 0.74 |
| 4 | 4.08                | −0.92         | 0.85 |

```
L₁ ≈ 0.55 + 0.64 + 0.74 + 0.85 = 2.78
```

**Loss went from 74 → 2.78 in one step.** That's a huge drop. Gradient descent works.

### Try γ = 0.1 (too big)

```
w₁ = 0 − 0.1·(−94) = 9.4
b₁ = 0 − 0.1·(−32) = 3.2
```

New predictions: `ŷₙ = 9.4·xₙ + 3.2`:

| n | ŷₙ   | eₙ    | eₙ²    |
|---|------|-------|--------|
| 1 | 12.6 | 10.6  | 112.4  |
| 2 | 22.0 | 19.0  | 361.0  |
| 3 | 31.4 | 26.4  | 697.0  |
| 4 | 40.8 | 34.8  | 1211.0 |

```
L₁ = 112.4 + 361 + 697 + 1211 ≈ 2381
```

**Loss went from 74 → 2381.** We overshot massively and landed far worse than we started.

### 🔑 Key observation

Too-small `γ` = slow but safe.
Too-large `γ` = **divergence**. Loss goes up, not down.

Section 6 will tell us *exactly* where this ceiling is.

---

## SECTION 6 — THE DIVERGENCE CEILING (WHY γ=0.1 BLEW UP)

### The short answer

There is a specific number `γ_max` above which gradient descent diverges. For a quadratic loss, this ceiling is:

```
γ_max = 2 / λ_max(H)
```

where `H` is the **Hessian** (matrix of second derivatives) and `λ_max` is its largest eigenvalue.

Let's compute `H` and see.

### Computing the Hessian

The Hessian is the matrix of second partial derivatives:

```
H = [ ∂²L/∂w²   ∂²L/∂w∂b ]
    [ ∂²L/∂b∂w ∂²L/∂b²   ]
```

From the gradient:
- `∂L/∂w = 2·Σ (w·xₙ + b − yₙ)·xₙ`
- `∂L/∂b = 2·Σ (w·xₙ + b − yₙ)`

Differentiate again:

```
∂²L/∂w²  = 2·Σ xₙ²     = 2·(1 + 4 + 9 + 16) = 2·30 = 60
∂²L/∂b²  = 2·Σ 1       = 2·4                         = 8
∂²L/∂w∂b = 2·Σ xₙ      = 2·(1+2+3+4) = 2·10          = 20
```

So:

```
H = [ 60  20 ]
    [ 20   8 ]
```

Notice: the Hessian does **not** depend on `w` or `b`. It's a constant matrix. That's because this loss is a pure quadratic.

### Computing the eigenvalues of H

Solve `det(H − λI) = 0`:

```
det [ 60−λ  20   ] = (60−λ)(8−λ) − 400
    [ 20    8−λ  ]

  = 480 − 60λ − 8λ + λ² − 400
  = λ² − 68λ + 80 = 0
```

Quadratic formula:

```
λ = (68 ± √(68² − 320)) / 2
  = (68 ± √4304) / 2
  ≈ (68 ± 65.6) / 2

λ_max ≈ 66.8
λ_min ≈ 1.2
```

### The divergence ceiling

```
γ_max = 2 / λ_max ≈ 2 / 66.8 ≈ 0.030
```

**Verify:**
- `γ = 0.01` is below `0.030` → converges ✓
- `γ = 0.1` is above `0.030` → diverges ✓

The 1D theory predicted this. It matches the experiment.

### 🔑 Key observation

| γ range         | behavior           |
|-----------------|--------------------|
| `γ < 1/λ_max`   | smooth descent     |
| `γ = 1/λ_max`   | optimal for a single axis |
| `1/λ_max < γ < 2/λ_max` | zigzags but converges |
| `γ ≥ 2/λ_max`   | diverges           |

If you've ever watched a neural net "blow up to NaN in 2 steps" — you crossed the `2/λ_max` threshold.

---

## SECTION 7 — THE CONDITION NUMBER (WHY GRADIENT DESCENT ZIGZAGS)

### Definition

```
κ = λ_max / λ_min
```

For our problem:

```
κ = 66.8 / 1.2 ≈ 56
```

### What κ means geometrically

Imagine the contour lines of our loss (sets where `L = constant`). The Hessian eigenvectors are the **axes** of those elliptical contours. The eigenvalues are the **curvatures** along those axes.

- `κ = 1` → circles. Gradient points directly at the minimum.
- `κ = 56` → long thin ellipses, ~√56 ≈ 7.5 times longer in one direction than the other. Gradient points roughly sideways, not at the minimum.

This is why gradient descent **zigzags**.

### Why κ also limits speed

Even with the best possible `γ`, the error shrinks by at most this factor per step:

```
error decay factor  =  (κ − 1) / (κ + 1)
```

| κ    | decay factor | progress per step |
|------|--------------|-------------------|
| 1    | 0            | done in 1 step    |
| 10   | 0.82         | 18% improvement   |
| 56   | 0.965        | 3.5% improvement  |
| 100  | 0.98         | 2% improvement    |
| 10⁶  | 0.999998     | essentially zero  |

**Our κ = 56 problem needs roughly 100+ steps to get near the optimum. κ is the bottleneck.**

### 🚨 Trap

"Just lower the learning rate to stop zigzag." **No.** 

Lowering `γ` makes each step smaller, but you still zigzag — just more slowly. The zigzag is **geometric** (a property of the loss function), not dynamic (a property of γ). Fix requires either changing the *direction* (momentum, Newton) or changing the *problem* (feature scaling).

---

## SECTION 8 — FIXING THE CONDITION NUMBER BY SCALING FEATURES

### The idea

The Hessian entries depended on `Σ xₙ²` and `Σ xₙ`. If we change the data's scale, we change the Hessian.

Replace `xₙ` by its **standardized** version:

```
x̃ₙ = (xₙ − x̄) / σₓ
```

where `x̄ = mean(x)` and `σₓ = std(x)`.

For our data:
- `x̄ = (1+2+3+4)/4 = 2.5`
- `σₓ = √(var) = √1.25 ≈ 1.118`

New `x̃`:

| n | xₙ  | x̃ₙ    |
|---|-----|--------|
| 1 | 1   | −1.34  |
| 2 | 2   | −0.45  |
| 3 | 3   |  0.45  |
| 4 | 4   |  1.34  |

Now `Σ x̃ₙ = 0` and `Σ x̃ₙ² = 4`. The new Hessian becomes:

```
H̃ = [ 2·4   2·0 ] = [ 8  0 ]
    [ 2·0   2·4 ]   [ 0  8 ]
```

**Diagonal. Both eigenvalues equal 8. κ = 1.**

### 🔑 Key observation

Contours are now perfect circles. Gradient descent with any reasonable `γ` goes **straight to the optimum**, no zigzag, in very few steps.

This is why we **standardize features** before training linear/logistic regression or any non-deep model. Not vague "it helps" — it literally makes `κ = 1` instead of 56.

For deep nets, **batch normalization** / **layer normalization** play the same role for intermediate layers.

---

## SECTION 9 — MOMENTUM (equations 7.11, 7.12)

### The update rule

```
Δᵢ = new move at step i = (wᵢ − wᵢ₋₁, bᵢ − bᵢ₋₁)

[ wᵢ₊₁ ]   [ wᵢ ]         [ ∂L/∂w ]       [ Δᵢ components ]
[ bᵢ₊₁ ] = [ bᵢ ]  − γ  · [ ∂L/∂b ]  + α· [              ]
```

| symbol | meaning |
|--------|---------|
| `Δᵢ`   | the actual move we made last step |
| `α`    | momentum coefficient, in `[0, 1]` |
| `γ`    | step size, same as before |

### Trace it by hand

Start at `(w₀, b₀) = (0, 0)`, `Δ₀ = (0, 0)`, `γ = 0.01`, `α = 0.9`.

**Step 1:** gradient at (0,0) is `(−94, −32)` (from Section 4).

```
w₁ = 0 − 0.01·(−94) + 0.9·0 = 0.94
b₁ = 0 − 0.01·(−32) + 0.9·0 = 0.32
Δ₁ = (0.94, 0.32)
```

Same as plain GD because `Δ₀ = 0`. OK.

**Step 2:** gradient at `(0.94, 0.32)`. Compute residuals:

| n | eₙ = 0.94·xₙ + 0.32 − yₙ |
|---|----|
| 1 | −0.74 |
| 2 | −0.80 |
| 3 | −0.86 |
| 4 | −0.92 |

```
∂L/∂w = 2·[(−0.74)(1) + (−0.80)(2) + (−0.86)(3) + (−0.92)(4)]
       = 2·[−0.74 − 1.60 − 2.58 − 3.68]
       = 2·(−8.60) = −17.2

∂L/∂b = 2·(−0.74 − 0.80 − 0.86 − 0.92) = 2·(−3.32) = −6.64
```

Now apply the momentum update:

```
w₂ = 0.94  − 0.01·(−17.2)  + 0.9·(0.94)
   = 0.94  + 0.172         + 0.846
   = 1.958

b₂ = 0.32  − 0.01·(−6.64)  + 0.9·(0.32)
   = 0.32  + 0.0664        + 0.288
   = 0.674

Δ₂ = (1.958 − 0.94, 0.674 − 0.32) = (1.018, 0.354)
```

### Compare: WITHOUT momentum (α = 0)

```
w₂ = 0.94 + 0.172  = 1.112
b₂ = 0.32 + 0.0664 = 0.386
```

Move from step 1 to step 2 was only `(0.172, 0.0664)`.

**With momentum, the step-2 move was (1.018, 0.354) — about 6× bigger.** That's the acceleration, exactly because both gradients (step 1 and step 2) pointed in the same general direction: momentum added them up.

### Unroll the recursion

```
Δᵢ  ≈  − γ·gᵢ₋₁  − αγ·gᵢ₋₂  − α²γ·gᵢ₋₃  − α³γ·gᵢ₋₄  − ...
```

The current move is an **exponentially weighted average** of all past gradients. Weights `1, α, α², α³, ...` decay geometrically.

### Effective step size

If every gradient is the same (we're heading down a valley):

```
sum of weights = 1 + α + α² + ... = 1 / (1 − α)
```

| α    | effective multiplier |
|------|----------------------|
| 0    | 1 (no momentum)     |
| 0.5  | 2                   |
| 0.9  | 10                  |
| 0.99 | 100                 |

So `α = 0.9` makes each step effectively **10× bigger** along consistent directions.

### Why it kills zigzag

When gradients alternate direction (back-and-forth across a narrow valley), `gᵢ ≈ −gᵢ₋₁`, so consecutive weighted terms cancel. The sideways motion shrinks. The forward motion survives and compounds.

### 🚨 Traps

- **"α is just another learning rate."** No. γ sets step magnitude; α sets memory length. They interact, but they are not interchangeable.
- **"More momentum is always better."** No. At α = 0.99, consistent directions get a 100× boost — you overshoot the minimum and oscillate forever.
- **"Momentum requires a convex loss."** No. It's used in every deep-learning optimizer.

### Where this appears in ML

`torch.optim.SGD(params, lr=γ, momentum=α)` — equations (7.11), (7.12) verbatim.

---

## SECTION 10 — STOCHASTIC GRADIENT DESCENT

### The motivation

The full gradient is a sum over N points. If N is 10 million, that's expensive. Can we estimate the gradient from a **subset**?

### The key fact: the full gradient is a sum

Recall:

```
∇L = Σ ∇Lₙ     (sum over n = 1..N)
```

Divide both sides by N:

```
∇L / N = (1/N)·Σ ∇Lₙ = average of per-point gradients
```

The average of `∇Lₙ` over the dataset is `(1/N)·∇L`. That's a **population mean**.

### Unbiased estimate

If we pick a random `n` uniformly from `{1, 2, 3, 4}`, the expected per-point gradient equals the population mean:

```
E[ ∇Lₙ ] = (1/N)·Σ ∇Lₙ = ∇L / N
```

So `N·∇Lₙ` is an **unbiased estimator** of `∇L` — on average it equals the true gradient.

### Compute per-point gradients at (0, 0)

From the table in Section 4:

| n | ∇Lₙ at (0,0) = (2·eₙ·xₙ, 2·eₙ) |
|---|--------------------------------|
| 1 | (−4, −4)                       |
| 2 | (−12, −6)                      |
| 3 | (−30, −10)                     |
| 4 | (−48, −12)                     |
|   | sum = (−94, −32)               |

Mean = sum/4 = (−23.5, −8). That's `∇L / 4`.

### SGD update with one random point

Pick `n = 2` (random choice). Use `∇L₂ = (−12, −6)`.

Two conventions:

**Convention A (what MML writes):** use `∇Lₙ` directly, but scale by N:
```
w₁ = 0 − γ · N · (−12) = 0 − 0.01 · 4 · (−12) = 0.48
```

**Convention B (what PyTorch does):** use the *mean* loss, and use `∇Lₙ` directly with no N-scaling:
```
w₁ = 0 − γ · (−12) = 0 − 0.01 · (−12) = 0.12
```

Both are valid; they differ only in how you define the loss (sum vs mean). Most ML code uses Convention B (mean loss), so γ doesn't depend on the dataset size.

### Compare to full-batch

At step 0:
- Full-batch direction: (−94, −32) ≈ (3:1 ratio of w to b components)
- Single point n=2: (−12, −6) = (2:1 ratio)
- Single point n=4: (−48, −12) = (4:1 ratio)

Each individual gradient is **noisy** — different point, different direction. But **on average**, they equal the true gradient.

### 🔑 Key observation

**Each single-point gradient is wrong. Their average is right.** SGD exploits this: many cheap, noisy, unbiased steps instead of few expensive exact steps.

---

## SECTION 11 — MINI-BATCHES (REDUCING VARIANCE)

### The idea

Instead of using 1 point (noisy) or all N points (expensive), use a **mini-batch** of size B.

### Mini-batch of size 2 — pick points n = 2 and n = 3

```
ĝ = (1/2) · [ ∇L₂ + ∇L₃ ]
  = (1/2) · [ (−12, −6) + (−30, −10) ]
  = (1/2) · (−42, −16)
  = (−21, −8)
```

Compare to the true mean `(−23.5, −8)`. Much closer than any single point. Much cheaper than all 4.

### Variance of the estimator

For a random subset of size B:

```
Var(estimator) ∝ 1/B
```

| B | variance scale | noise scale (std) |
|---|----------------|-------------------|
| 1   | 1        | 1.00 |
| 4   | 1/4      | 0.50 |
| 16  | 1/16     | 0.25 |
| 64  | 1/64     | 0.13 |
| 256 | 1/256    | 0.06 |

### Diminishing returns

Standard deviation scales as `1/√B`. Going from B=1 to B=16 cuts noise by 4×. Going from B=256 to B=1024 cuts it by only 2×. **Doubling B eventually stops buying much.**

### The practical sweet spot

| B       | trade-off |
|---------|-----------|
| 1       | maximum noise, cheapest per step, may not use GPU fully |
| 32–256  | good noise/cost balance, fits GPU well (common default) |
| full N  | no noise, but slow per step, may not fit in memory |

### 🔑 Key observation

Mini-batch SGD is the best of both worlds: low enough cost per step to be fast, high enough batch to be stable, small enough batch to be noisy in a *useful* way (next section).

---

## SECTION 12 — NOISE IS A FEATURE, NOT A BUG

### The observation

Full-batch GD is **deterministic**. Same start → same path → same endpoint. If there's a bad local minimum near the starting point, GD falls into it and stays.

SGD is **stochastic**. Each step is a noisy estimate, so the path wiggles. This wiggling lets the iterate **escape** shallow bad regions.

### What this means for non-convex problems

On our linear regression, there's only one minimum — noise is not needed. But on a neural net:

- **Saddle points:** at a saddle, the gradient is small but the point is not a minimum. Full-batch GD crawls through a saddle slowly. SGD's noise kicks it off the saddle in a random direction fast.
- **Sharp minima vs flat minima:** sharp minima (narrow dip) are unstable under noise — small perturbations push you out. Flat minima (wide basin) tolerate noise. So SGD **preferentially settles in flat minima**.
- **Flat minima generalize better.** A flat minimum means "many nearby parameter values give similar loss" → small shifts (from training to test distribution) don't hurt much. Sharp minima generalize poorly.

### 🔑 Key observation

SGD is **not** a worse version of GD. It's a different algorithm with a *different bias* — toward flat, well-generalizing minima. That's why it's the default for deep learning, not just because it's faster.

---

## SECTION 13 — LEARNING RATE DECAY (SO SGD ACTUALLY CONVERGES)

### The problem

At the true minimum, the full gradient is zero, but the **mini-batch** gradient is still noisy. With a constant γ, you bounce around the minimum forever — a "noise ball" of radius proportional to γ.

### The fix

Shrink γ over time:

```
γᵢ decreases as i grows
```

Classical conditions (Robbins–Monro) for almost-sure convergence:

```
Σ γᵢ = ∞           (steps add up to enough distance to reach the minimum)
Σ γᵢ² < ∞          (variance is eventually controlled)
```

Example: `γᵢ = 1/i`. The harmonic series diverges (enough total motion), but the sum of squares converges (noise dies).

### In practice

Deep learning schedulers don't strictly follow Robbins–Monro but follow the same spirit:

| schedule | formula | intuition |
|----------|---------|-----------|
| Step decay | γ halves every K epochs | periodic cooling |
| Cosine | γ smoothly decays like a cosine wave | smooth cooling |
| Warmup + decay | γ rises, then decays | careful start + cooling |

### 🚨 Trap

"Training plateaued — must be stuck in a local minimum." Often wrong. The learning rate was too large for the current region of the loss surface, and you were bouncing. **Decay γ first, then diagnose.**

---

## SECTION 14 — PUTTING IT ALL TOGETHER: THE FULL ALGORITHM

This is what `torch.optim.SGD(params, lr=γ, momentum=α)` combined with a `DataLoader` and a `lr_scheduler` actually does:

```
initialize (w, b) randomly (or zeros for convex problems)
initialize Δ = (0, 0)
for i = 0, 1, 2, ...:
    1. sample a mini-batch B of size B from the dataset
    2. compute stochastic gradient:  ĝ = (1/B) · Σ ∇Lₙ  for n in B
    3. update parameters:  (w, b) ← (w, b) − γᵢ·ĝ + α·Δ
    4. store move:  Δ ← new (w,b) − old (w,b)
    5. decay γᵢ according to schedule
```

Every line maps to something you now understand:

| line | concept |
|------|---------|
| 1 | mini-batch sampling (§7.1.3) |
| 2 | unbiased gradient estimator (§7.1.3) |
| 3 | gradient descent update (§7.1) + momentum (§7.1.2) |
| 3 | γᵢ must stay below `2/λ_max` to avoid divergence (§7.1.1) |
| 3 | condition number κ determines how many iterations this needs (§7.1.1) |
| 4 | bookkeeping for momentum term |
| 5 | learning rate decay — lets SGD converge exactly (§7.1.3 Remark) |

---

## SECTION 15 — VERIFY IN NUMPY

Type this up. Your hand calculations (w₁, b₁, w₂, b₂) should match the printed values exactly.

```python
import numpy as np

# Data: column of ones for the bias
X = np.array([[1, 1],
              [2, 1],
              [3, 1],
              [4, 1]], dtype=float)
y = np.array([2, 3, 5, 6], dtype=float)

theta = np.zeros(2)   # theta[0] = w, theta[1] = b
delta = np.zeros(2)
gamma, alpha = 0.01, 0.9

def loss(theta):
    return np.sum((X @ theta - y) ** 2)

def full_grad(theta):
    r = X @ theta - y             # residuals
    return 2 * X.T @ r            # gradient

# ---- 1) plain GD, γ = 0.01 ----
t = np.zeros(2)
for i in range(3):
    g = full_grad(t)
    print(f"step {i}: theta={t}, loss={loss(t):.4f}, grad={g}")
    t = t - gamma * g

# ---- 2) plain GD, γ = 0.1 (DIVERGES) ----
t = np.zeros(2)
for i in range(3):
    g = full_grad(t)
    print(f"step {i} (big γ): theta={t}, loss={loss(t):.4f}")
    t = t - 0.1 * g

# ---- 3) GD with momentum ----
t = np.zeros(2); d = np.zeros(2)
for i in range(3):
    g = full_grad(t)
    t_new = t - gamma * g + alpha * d
    d = t_new - t
    t = t_new
    print(f"momentum step {i}: theta={t}, loss={loss(t):.4f}")

# ---- 4) Closed-form (exact answer) ----
theta_star = np.linalg.solve(X.T @ X, X.T @ y)
print("exact solution (normal equations):", theta_star)
# should be approximately [1.4, 0.5]

# ---- 5) Verify Hessian & eigenvalues ----
H = 2 * X.T @ X
print("Hessian:\n", H)
eigvals = np.linalg.eigvalsh(H)
print("eigenvalues:", eigvals)
print("condition number κ:", eigvals.max() / eigvals.min())
print("divergence threshold γ_max =", 2 / eigvals.max())
```

Expected output hints:
- After step 1 of GD with γ=0.01: `theta ≈ [0.94, 0.32]`
- After step 1 with γ=0.1: `theta = [9.4, 3.2]` and loss EXPLODES
- Eigenvalues of H: approximately `[1.2, 66.8]`
- κ ≈ 56
- γ_max ≈ 0.030 → our γ=0.01 is safe, γ=0.1 is past the ceiling
- Normal equation solution: `[1.4, 0.5]`

---

## SECTION 16 — CONCEPT CHECKLIST

By the time you've written this by hand, check off every one of these:

- [ ] **Sum-over-data loss** — L = Σ Lₙ, equation (7.13)
- [ ] **Gradient of a sum = sum of gradients** — why SGD is even possible
- [ ] **Per-point gradient derivation** — chain rule applied to squared error
- [ ] **GD update rule** — (w,b) ← (w,b) − γ·∇L, equation (7.6)
- [ ] **Why too-small γ is slow** — each step moves too little
- [ ] **Why too-large γ diverges** — overshoots the minimum
- [ ] **Hessian H** — matrix of second derivatives
- [ ] **Divergence ceiling γ_max = 2/λ_max** — exact, not heuristic
- [ ] **Eigenvalues of H** — curvatures along the principal axes
- [ ] **Condition number κ = λ_max/λ_min** — ratio of worst to best curvature
- [ ] **Zigzag explanation** — gradient perpendicular to elongated contours
- [ ] **Feature scaling fixes κ** — standardization makes the Hessian closer to identity
- [ ] **Momentum update rule** — equations (7.11), (7.12)
- [ ] **Effective step multiplier 1/(1−α)** — ~10× when α=0.9
- [ ] **Momentum cancels zigzag, accelerates valley motion** — signal/noise argument
- [ ] **SGD = stochastic estimate of the true gradient** — unbiased
- [ ] **Mini-batch variance scales as 1/B** — diminishing returns
- [ ] **Noise as implicit regularization** — flat vs sharp minima
- [ ] **Learning rate decay** — required for SGD to converge exactly
- [ ] **Full algorithm = mini-batch SGD with momentum** — same loop as `torch.optim.SGD`

---

## FINAL NOTE

Every optimizer you will ever use — SGD, Adam, RMSProp, L-BFGS, whatever — is a variation on the pieces in this example. Different direction (momentum, preconditioning), different estimate (mini-batch, variance-reduced), different step size (decay, adaptive), but the skeleton is the same.

The 4 data points of this example are enough to see the whole picture. When you write it by hand, you are literally computing the same quantities that PyTorch computes on every training step of a transformer with 100 billion parameters — only with different numbers.

---
