### Overview
Deep feedforward networks (often called multilayer perceptrons) are parametric function families that compute a prediction by repeatedly applying linear maps and nonlinear elementwise transforms. They implement a learned feature map $\varphi(x; \theta)$ by composing many simple layers, then apply a final linear readout $w$ to produce outputs. The learning algorithm adjusts $\theta$ (and $w$) so $\varphi$ becomes a representation that makes the prediction task easy.

---

### Model, notation, and core formulas

- **Input**: $x \in \mathbb{R}^d$ (raw data vector).
- **Layers**: indexed by $\ell = 1, \ldots, L$. Each layer $\ell$ has weights $W^{[\ell]}$ and biases $b^{[\ell]}$.
  - $W^{[\ell]} \in \mathbb{R}^{n_{\ell} \times n_{\ell-1}}$ maps layer $(\ell-1)$ activations to layer $\ell$ pre-activations.
  - $b^{[\ell]} \in \mathbb{R}^{n_{\ell}}$ is the bias vector for layer $\ell$.
  - $n_0 = d$ (input size), $n_L = m$ (size of final representation $\varphi$).
- **Activation function**: $g^{[\ell]}(\cdot)$ applied elementwise (ReLU, tanh, sigmoid, etc.).
- **Hidden activations**: $h^{[0]} = x$. For $\ell = 1, \ldots, L$ compute
  $$
  a^{[\ell]} = W^{[\ell]} h^{[\ell-1]} + b^{[\ell]}, \qquad
  h^{[\ell]} = g^{[\ell]}\bigl(a^{[\ell]}\bigr).
  $$ 
  - $a^{[\ell]}$ are pre-activations (linear scores) at layer $\ell$.
  - $h^{[\ell]}$ are post-activation outputs (features) of layer $\ell$.
- **Representation (feature map)**: define $\varphi(x; \theta) = h^{[L]}$ where $\theta$ collects all weights and biases $\{W^{[\ell]}, b^{[\ell]}\}_{\ell=1}^L$.
- **Readout and prediction**:
  - Linear readout weights $w \in \mathbb{R}^{k \times m}$ ($k =$ output dimension); bias $c \in \mathbb{R}^k$.
  - Score vector $z = w\, \varphi(x; \theta) + c$.
  - Output $\hat{y} = \text{link}(z)$ (identity for regression; softmax for multiclass; sigmoid for binary).
- **Full model compactly**:
  $$
  f(x;\theta,w) = \text{link}\bigl(w\,\varphi(x;\theta)+c\bigr), \qquad
  \varphi(x;\theta) = h^{[L]}.
  $$

Intuition for symbols:
- $W^{[\ell]}$ linearly recombines previous features; $b^{[\ell]}$ shifts those linear combinations.
- $g^{[\ell]}$ injects nonlinearity so composition of layers is not just another linear map.
- $a^{[\ell]}$ are the raw linear signals the layer computes; $h^{[\ell]}$ are the "features" the next layer sees.
- $\theta$ = collection of all layer parameters; $w$ is the simple final classifier/regressor on top of $\varphi$.

---

### What φ(x; θ) does — role and intuition
- $\varphi$ is a mapping $x \to$ vector of learned features. Each coordinate $\varphi_j(x; \theta)$ is a scalar feature computed by composing many simple transforms.
- Intuition by layers:
  - Early layers detect local/simple patterns (edges, frequencies, n-grams).
  - Middle layers combine simple patterns into motifs or parts (shapes, syllables, phrases).
  - Late layers produce abstract, task-relevant features where classes or outputs separate linearly.
- Mathematical view: $\varphi$ maps raw input space into a representation space where the final readout $w$ can implement a simple decision boundary (often a hyperplane) or linear regression. The network learns coordinates of that space so that class-conditional distributions become separable or targets become linear functions of $\varphi$.
- Expressive power: composing linear maps without nonlinearity collapses to a single linear transform. Nonlinear $g$ at each layer allows the composition to represent highly nonlinear functions of $x$ while staying linear in the final $w$.

---

### How deep learning finds φ — loss, gradients, and optimization
- **Objective**: choose $\theta$ and $w$ to minimize empirical loss over dataset $\{(x^{(i)}, y^{(i)})\}$:
  $$
  L(\theta,w)=\frac{1}{N}\sum_{i=1}^N \ell\bigl(f(x^{(i)};\theta,w),\ y^{(i)}\bigr) + R(\theta,w),
  $$
  where $\ell$ is per-example loss (cross-entropy, squared error) and $R$ is regularization.
- **Gradient-based learning**:
  - Compute forward pass to get $\varphi(x^{(i)};\theta)$ and outputs.
  - Compute loss and then backward pass (backpropagation) to obtain gradients $\partial L/\partial w$ and $\partial L/\partial \theta$. Backprop uses chain rule through layers:
    $$
    \frac{\partial L}{\partial W^{[\ell]}} = \frac{\partial L}{\partial a^{[\ell]}}\, (h^{[\ell-1]})^\top, \qquad
    \frac{\partial L}{\partial b^{[\ell]}} = \frac{\partial L}{\partial a^{[\ell]}}.
    $$
    and
    $$
    \frac{\partial L}{\partial h^{[\ell-1]}} = (W^{[\ell]})^\top \frac{\partial L}{\partial a^{[\ell]}}.
    $$
    with elementwise factor for activation: $\partial L/\partial a^{[\ell]} = \partial L/\partial h^{[\ell]} \odot g^{[\ell]'}(a^{[\ell]})$.
  - Update parameters using an optimizer (SGD, SGD+momentum, Adam, RMSprop):
    $$
    	heta \leftarrow \theta - \eta \,\widehat{\nabla_\theta L}, \qquad
    w \leftarrow w - \eta \,\widehat{\nabla_w L},
    $$
    where $\eta$ is learning rate and hats denote gradients estimated on a mini-batch.
- **Why backprop finds $\varphi$**: gradients tell each parameter how changing it will change the final loss via its effect on downstream features; repeated small adjustments steer $\theta$ so $\varphi$ extracts features that reduce loss.
- **Practical optimizer choices**: mini-batch SGD for scalability; momentum to accelerate along consistent directions; adaptive learning rates to handle differing gradient scales; weight decay ($L_2$) as regularizer.

---

### Why and when we introduce nonlinearity and depth
- Single linear layer: if $\varphi(x) = A x + b$ (affine), then $w^T \varphi(x) = (A^T w)^T x + w^T b$ — overall still linear in $x$. Depth with only linear maps collapses to single linear transform; no extra representational power.
- Nonlinearity $g$ breaks the collapse: composition of linear maps and nonlinearities yields functions beyond linear span of inputs. Example: using $\varphi(x) = [x_1^2, x_1 x_2]$ cannot be represented by any single linear map of $x$.
- When to go nonlinear and deep:
  - When the target function $f^*$ depends on interactions or hierarchical structure in $x$ that a single linear mapping cannot capture.
  - When handcrafted $\varphi$ or kernel methods are impractical or fail to scale. Deep nonlinear $\varphi$ is chosen when data is abundant and tasks require learning complex, hierarchical invariances (images, audio, language).
- Intuition of depth: each layer re-represents the input, creating new coordinates; depth allows exponentially many distinct features with modest parameters, often matching the compositional structure of natural signals.

---

### Additional practical intuitions and book’s teaching points
- Separation of concerns: learnable $\varphi$ ($\theta$) builds representation; simple linear $w$ reads that representation. This clarifies why networks generalize: they learn features useful across examples rather than memorizing raw inputs.
- Nonconvexity tradeoff: learning $\theta$ yields nonconvex optimization, losing global guarantees but gaining much greater representational power; in practice stochastic gradient methods find useful minima.
- Inductive bias via architecture: convolutional layers encode locality and translation equivariance; recurrence or attention encodes sequence structure; these inductive biases make $\varphi$ easier to learn and generalize better from less data.
- Regularization and training recipes matter: batch normalization, dropout, data augmentation, careful initialization, and learning-rate schedules all shape how $\varphi$ is learned and how well features generalize.
- Interpretability: $\varphi$'s coordinates can often be inspected to see what features a network uses, but they are learned and distributed (not single human-named features).

---

### Concise summary
A deep feedforward network parametrizes $\varphi(x; \theta)$ by composing linear maps and nonlinear activations; $\varphi$ converts raw inputs into a learned representation where a linear readout $w$ can perform the task. Learning $\varphi$ uses backpropagation and gradient-based optimization to adjust all layer parameters so that the representation becomes task-useful. Nonlinearity at each layer is essential: without it depth collapses to a single linear map. The deep-learning approach trades convexity for representation learning, using architecture and optimization tricks to reliably find good $\varphi$ in practice, which is the central lesson the book emphasizes.