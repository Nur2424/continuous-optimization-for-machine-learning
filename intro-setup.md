---

## What these pages are really doing

They're setting up a **single, central claim**:

> *Training an ML model = solving $\min_x f(x)$ where $f$ is differentiable. In general, we can't solve it analytically. So we iterate.*

Everything else in the chapter flows from this. Let me break out the sub-ideas.

---

## 1. Why "minimization" is the default convention

**🧠 Intuition:** The book says "objective functions are intended to be minimized." This is a *convention*, not a mathematical necessity.

- Loss functions (MSE, cross-entropy) → you want these **small** → minimize.
- Likelihoods, rewards → you want these **big** → but we flip the sign and call it "negative log-likelihood" or "negative reward" so we can still minimize.

**Why ML cares:** every optimizer in PyTorch (`torch.optim.SGD`, `Adam`, etc.) is a **minimizer**. That's why you compute `loss = -log_likelihood` and then `loss.backward()`. If you ever see `maximize=True` flag on an optimizer, it's just internally negating the gradient.

**🚨 Trap:** People sometimes write "we want to maximize accuracy" and then try to backprop through it. Accuracy isn't differentiable. That's why the *loss* (cross-entropy, a differentiable surrogate) is what gets minimized, not accuracy itself.

---

## 2. The differentiability assumption

The book says: *"we will assume that our objective function is differentiable, hence we have access to a gradient at each location."*

**This is a massive assumption, and ML abuses it constantly.**

- ReLU is not differentiable at 0. PyTorch just picks a subgradient (returns 0) and ships it.
- L1 regularization ($\lVert w \rVert_1$) is not differentiable where any $w_i = 0$. Same trick.
- Indicator functions (like accuracy) aren't differentiable at all — hence we use surrogates.

**⚠️ Important:** The entire gradient-descent machine rests on this assumption. When it almost-fails (ReLU), we paper over it. When it fails hard (accuracy, 0-1 loss), we replace the objective with a smooth surrogate.

**Where this appears in ML:** every time you wonder "why does the loss have this specific form instead of what I actually care about?" — it's because we need a differentiable objective.

---

## 3. Gradient = uphill. Negative gradient = downhill.

**🧠 Intuition:** The gradient $\nabla f(x)$ points in the direction of **steepest *increase*** of $f$. Not "toward the minimum." Not "toward the optimum." **Steepest local increase.**

In 1D this is just the sign of the derivative:
- $f'(x) > 0$: function going up → move left (negative direction) to decrease
- $f'(x) < 0$: function going down (to the right) → move right (positive direction) to decrease

Either way: **move opposite the gradient**. That's why the book shows arrows pointing downhill — those are $-\nabla f$.

**🚨 Trap — and this matters for neural nets:** the negative gradient points toward *lower* $f$, **not toward the minimum**. It's a *local* direction. It knows nothing about where the minimum is. It might be pointing you toward a local min, a saddle, or off a cliff.

---

## 4. Stationary points and the second derivative test

The book's example: $\ell(x) = x^4 + 7x^3 + 5x^2 - 17x + 3$

**Step 1 — find stationary points:** set $\ell'(x) = 0$.

$$\ell'(x) = 4x^3 + 21x^2 + 10x - 17 = 0$$

This is cubic → generally 3 real roots. Book says they're approximately at $x \approx -4.5, -1.4, 0.7$.

**Step 2 — classify with $\ell''(x)$:**

$$\ell''(x) = 12x^2 + 42x + 10$$

| $x$ | $\ell''(x)$ | Sign | Verdict |
|------|------|------|------|
| $-4.5$ | $12(20.25) + 42(-4.5) + 10 = 64$ | $+$ | **min** ✓ |
| $-1.4$ | $12(1.96) + 42(-1.4) + 10 \approx -25.3$ | $-$ | **max** ✓ |
| $0.7$ | $12(0.49) + 42(0.7) + 10 \approx 45.3$ | $+$ | **min** ✓ |

**The rule in 1D:**
- $f''(x^*) > 0$ at stationary point → local **min** (curves up)
- $f''(x^*) < 0$ → local **max** (curves down)
- $f''(x^*) = 0$ → test is inconclusive (could be inflection)

**⚠️ Important — what changes in higher dimensions** (since this is your end goal):

In $\mathbb{R}^D$, the second derivative becomes the **Hessian matrix** $H = \nabla^2 f$. The test becomes:

- All eigenvalues of $H$ positive → local min
- All eigenvalues negative → local max
- **Mixed signs** → **saddle point** (new phenomenon, doesn't exist in 1D!)
- Any zero eigenvalue → inconclusive

> 🧠 **Saddle points are everywhere in deep learning.** In high dimensions, it's combinatorially unlikely that *all* eigenvalues of the Hessian share a sign. So most stationary points of a neural net's loss are saddles, not minima. This is one of those "doesn't generalize directly" warnings the book gives on page 227.

The book hasn't introduced this yet — I'm flagging it so you don't build a 1D mental model that breaks when you hit neural nets.

---

## 5. The key reason we need gradient descent at all

The book mentions **Abel–Ruffini**: no algebraic (closed-form) formula for roots of polynomials of degree 5+.

But the real reason is broader: **almost all ML objectives have no closed-form critical point**.

- Linear regression with MSE: closed form exists (normal equations). Rare exception.
- Ridge regression: closed form exists. Rare exception.
- Logistic regression: **no closed form**. Must iterate.
- SVM: **no closed form**. Must iterate (usually dual).
- Neural net: **absolutely no closed form**, not even close.

So we're forced to:
1. Pick a starting point $x_0$.
2. Follow $-\nabla f$ iteratively.
3. Hope.

That's gradient descent. Details in §7.1.

---

## 6. The starting-point problem (this is the big ML lesson on page 227)

The book's most important observation on this page, in my opinion:

> *"if we had started at the right side (e.g., $x_0 = 0$) the negative gradient would have led us to the wrong minimum."*

**🧠 Intuition:** Gradient descent is **local and greedy**. It only sees the slope *right under its feet*. It does not know a better valley exists elsewhere. Starting at $x_0 = 0$, the slope points right → you roll to the shallow minimum near $0.7$ and get stuck there. You never even *know* about the deep minimum at $-4.5$.

**🚨 Trap you probably have:** "Gradient descent finds the minimum." **NO.** Gradient descent finds **a** stationary point — usually a local min in the basin of whatever $x_0$ you started at. For non-convex problems, the initial point *determines the outcome*.

**Where this appears in ML:**
- **Why neural net initialization matters so much** (Xavier, He init). Bad init → bad basin → stuck.
- **Why we train the same model multiple times with different seeds** and compare.
- **Why warm-starting from a pretrained model is so powerful** — you're starting near a good basin already.
- **Why SGD's noise is a *feature*, not a bug** — it can bump you out of bad local minima. (Later section.)

---

## 7. Convexity preview (the book intentionally stops here — so I will too)

The book drops: *"For convex functions, all local minima are global minima."* And then promises §7.3.

I'll just say one thing and stop:

**⚠️ The ML world splits cleanly:**
- **Convex ML:** linear regression, logistic regression, SVM with convex loss, ridge, lasso. Local min = global min. Optimization is "easy mode."
- **Non-convex ML:** neural networks, GMM training, matrix factorization. Tons of local mins, saddles, flat regions. Optimization is a research field.

We'll develop this properly in §7.3. *This will make more sense when we get to convex optimization.*

---

## 8. The "1D → high-D" warning

The book ends page 227 with: *"some concepts do not generalize directly to higher dimensions, therefore some care needs to be taken when reading."*

**Concrete things that change going from 1D to $\mathbb{R}^D$:**

| Concept | 1D | Higher-D |
|------|------|------|
| Derivative | scalar $f'(x)$ | vector $\nabla f(x) \in \mathbb{R}^D$ |
| 2nd deriv | scalar $f''(x)$ | matrix $H \in \mathbb{R}^{D \times D}$ |
| Classification | sign of $f''$ | **eigenvalues** of $H$ |
| Stationary types | min, max, inflection | min, max, **saddle** |
| "Direction" | just $\pm 1$ | infinitely many directions |
| Escaping a min | need to cross a barrier | can sometimes go around it |

Keep this table in your head. A lot of 1D visual intuition carries — but the saddle and the "infinitely many directions" parts are genuinely new.

---

## Summary (1–2 sentences)

**What these pages actually said:** Training ML = minimizing a differentiable loss; in general we can't solve $\nabla f = 0$ analytically, so we iterate from some starting point following $-\nabla f$; for non-convex problems, the starting point determines which local min (if any) you land in.

---
