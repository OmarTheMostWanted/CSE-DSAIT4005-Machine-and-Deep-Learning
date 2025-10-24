### Overview
Deep feedforward networks (often called multilayer perceptrons) are parametric function families that compute a prediction by repeatedly applying linear maps and nonlinear elementwise transforms. They implement a learned feature map $\phi(x; \theta)$ by composing many simple layers, then apply a final linear readout $w$ to produce outputs. The learning algorithm adjusts $\theta$ (and $w$) so $\phi$ becomes a representation that makes the prediction task easy.

A linear map is a matrix multiplication plus an optional bias vector that recombines incoming numbers into new numbers using weighted sums. A nonlinear activation function (ReLU, tanh, sigmoid, etc.) is applied elementwise to inject nonlinearity, allowing the composition of layers to represent complex functions of the input.


### Model, notation, and core formulas

- **Input**: $x \in \mathbb{R}^d$ (raw data vector).
- **Layers**: indexed by $\ell = 1, \ldots, L$. Each layer $\ell$ has weights $W^{[\ell]}$ and biases $b^{[\ell]}$.
  - $W^{[\ell]} \in \mathbb{R}^{n_{\ell} \times n_{\ell-1}}$ is a weight matrix that maps layer $(\ell-1)$ activations to layer $\ell$ pre-activations.
    - Each row of $W^{[\ell]}$ picks a weighted sum of the previous layer's features, this is a new raw signal for the next layer.
  - $b^{[\ell]} \in \mathbb{R}^{n_{\ell}}$ is the bias vector for layer $\ell$.
  - $n_0 = d$ (input size), $n_L = m$ (size of final representation $\phi$).
- **Activation function**: $g^{[\ell]}(\cdot)$ applied elementwise (ReLU, tanh, sigmoid, etc.).
- **Hidden activations**: $h^{[0]} = x$. For $\ell = 1, \ldots, L$ compute
  $$
  a^{[\ell]} = W^{[\ell]} h^{[\ell-1]} + b^{[\ell]}, \qquad
  h^{[\ell]} = g^{[\ell]}\bigl(a^{[\ell]}\bigr).
  $$ 
  - $a^{[\ell]}$ are pre-activations (linear scores) at layer $\ell$.
  - $h^{[\ell]}$ are post-activation outputs (features) of layer $\ell$.
- **Representation (feature map)**: define $\phi(x; \theta) = h^{[L]}$ where $\theta$ collects all weights and biases $\{W^{[\ell]}, b^{[\ell]}\}_{\ell=1}^L$.
- **Readout and prediction**:
  - Linear readout weights $w \in \mathbb{R}^{k \times m}$ ($k =$ output dimension); bias $c \in \mathbb{R}^k$.
    - $w$ is a matrix that takes the activation vector of the last layer and maps it to output scores.
  - Score vector $z = w\, \phi(x; \theta) + c$.
    - $c$ is a bias vector added to the score vector.
  - Output $\hat{y} = \text{link}(z)$ (identity for regression; softmax for multiclass; sigmoid for binary).
    - link is the final activation function that maps scores to predictions.
- **Full model compactly**:
  $$
  f(x;\theta,w) = \text{link}\bigl(w\,\phi(x;\theta)+c\bigr), \qquad
  \phi(x;\theta) = h^{[L]}.
  $$

Intuition for symbols:
- $W^{[\ell]}$ linearly recombines previous features; $b^{[\ell]}$ shifts those linear combinations.
- $g^{[\ell]}$ injects nonlinearity so composition of layers is not just another linear map.
  - Without $g$, multiple layers collapse to a single linear transform, losing representational power.
  - Nonlinearity gives the network the ability to model complex, nonlinear functions that warp space in ways linear maps cannot.
- $a^{[\ell]}$ are the raw linear signals the layer computes; $h^{[\ell]}$ are the "features" the next layer sees.
- $\theta$ = collection of all layer parameters; $w$ is the simple final classifier/regressor on top of $\phi$.

---

### What $\phi(x; \theta)$ does — role and intuition
- $\phi$ is a mapping $x \to$ vector of learned features. Each coordinate $\phi_j(x; \theta)$ is a scalar feature computed by composing many simple transforms.
  - $\phi$ is literally the function you get by composing many simple functions: multiply by a matrix, add a bias, apply elementwise nonlinearity, repeat.  
- Intuition by layers:
  - Early layers detect local/simple patterns (edges, frequencies, n-grams).
  - Middle layers combine simple patterns into motifs or parts (shapes, syllables, phrases).
  - Late layers produce abstract, task-relevant features where classes or outputs separate linearly.
- Mathematical view: $\phi$ maps raw input space into a representation space where the final readout $w$ can implement a simple decision boundary (often a hyperplane) or linear regression. The network learns coordinates of that space so that class-conditional distributions become separable or targets become linear functions of $\phi$.
- Expressive power: composing linear maps without nonlinearity collapses to a single linear transform. Nonlinear $g$ at each layer allows the composition to represent highly nonlinear functions of $x$ while staying linear in the final $w$.

---

### How deep learning finds $\phi$ — loss, gradients, and optimization
- **Objective**: choose $\theta$ and $w$ to minimize empirical loss over dataset $\{(x^{(i)}, y^{(i)})\}$:
  $$
  L(\theta,w)=\frac{1}{N}\sum_{i=1}^N \ell\bigl(f(x^{(i)};\theta,w),\ y^{(i)}\bigr) + R(\theta,w),
  $$
  where $\ell$ is per-example loss (cross-entropy, squared error) and $R$ is regularization.
- **Gradient-based learning**:
  - Compute forward pass to get $\phi(x^{(i)};\theta)$ and outputs.
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
- **Why backprop finds $\phi$**: gradients tell each parameter how changing it will change the final loss via its effect on downstream features; repeated small adjustments steer $\theta$ so $\phi$ extracts features that reduce loss.
- **Practical optimizer choices**: mini-batch SGD for scalability; momentum to accelerate along consistent directions; adaptive learning rates to handle differing gradient scales; weight decay ($L_2$) as regularizer.

---

### Why and when we introduce nonlinearity and depth
- Single linear layer: if $\phi(x) = A x + b$ (affine), then $w^T \phi(x) = (A^T w)^T x + w^T b$ — overall still linear in $x$. Depth with only linear maps collapses to single linear transform; no extra representational power.
- Nonlinearity $g$ breaks the collapse: composition of linear maps and nonlinearities yields functions beyond linear span of inputs. Example: using $\phi(x) = [x_1^2, x_1 x_2]$ cannot be represented by any single linear map of $x$.
- When to go nonlinear and deep:
  - When the target function $f^*$ depends on interactions or hierarchical structure in $x$ that a single linear mapping cannot capture.
  - When handcrafted $\phi$ or kernel methods are impractical or fail to scale. Deep nonlinear $\phi$ is chosen when data is abundant and tasks require learning complex, hierarchical invariances (images, audio, language).
- Intuition of depth: each layer re-represents the input, creating new coordinates; depth allows exponentially many distinct features with modest parameters, often matching the compositional structure of natural signals.

---

### Additional practical intuitions and book’s teaching points
- Separation of concerns: learnable $\phi$ ($\theta$) builds representation; simple linear $w$ reads that representation. This clarifies why networks generalize: they learn features useful across examples rather than memorizing raw inputs.
- Nonconvexity tradeoff: learning $\theta$ yields nonconvex optimization, losing global guarantees but gaining much greater representational power; in practice stochastic gradient methods find useful minima.
- Inductive bias via architecture: convolutional layers encode locality and translation equivariance; recurrence or attention encodes sequence structure; these inductive biases make $\phi$ easier to learn and generalize better from less data.
- Regularization and training recipes matter: batch normalization, dropout, data augmentation, careful initialization, and learning-rate schedules all shape how $\phi$ is learned and how well features generalize.
- Interpretability: $\phi$'s coordinates can often be inspected to see what features a network uses, but they are learned and distributed (not single human-named features).

---

### Concise summary
A deep feedforward network parametrizes $\phi(x; \theta)$ by composing linear maps and nonlinear activations; $\phi$ converts raw inputs into a learned representation where a linear readout $w$ can perform the task. Learning $\phi$ uses backpropagation and gradient-based optimization to adjust all layer parameters so that the representation becomes task-useful. Nonlinearity at each layer is essential: without it depth collapses to a single linear map. The deep-learning approach trades convexity for representation learning, using architecture and optimization tricks to reliably find good $\phi$ in practice, which is the central lesson the book emphasizes.