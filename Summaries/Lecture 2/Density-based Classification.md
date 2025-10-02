The lecture slides contain numerous mathematical expressions relating primarily to probability, Bayes' theorem, parameter estimation for Gaussian distributions, and resulting discriminant functions used in density-based classification.

Below is an explanation of the formulas presented in the sources, grouped by topic:

---

### 1. Fundamental Probability and Bayes' Theorem

The core of density-based classification relies on finding the posterior probability $p(y|x)$. This probability represents the likelihood that an object belongs to class $y$, given its feature vector $x$.

#### Bayes' Rule
The key relationship used is **Bayes' theorem**, which relates the posterior probability to distributions that are often easier to estimate:

$$p(y|x) = \frac{p(x|y)p(y)}{p(x)} \quad$$

This formula breaks down the probability calculation into three components:

*   **$p(y|x)$**: The **class posterior (conditional) distribution**. This is the probability that an object belongs to class $y$, given the data $x$. In practice, we aim to find an approximation, denoted as $\hat{p}(y|x)$.
*   **$p(x|y)$**: The **class conditional distribution**. This is the density function of data $x$, given that it belongs to class $y$. Estimated versions are denoted $\hat{p}(x|y)$.
*   **$p(y)$**: The **class prior (unconditional) distribution**. This is the inherent probability of observing class $y$ regardless of the input data. Estimated versions are denoted $\hat{p}(y)$.
*   **$p(x)$**: The **data distribution** (or unconditional probability). This is the density function of observing the data $x$. Since $p(x)$ is independent of the class, it is often dropped when defining classifiers.

#### Estimated Probabilities and Priors
In practice, these probabilities are approximated ($\hat{p}$).

*   **Estimated Class Prior**:
    $$\hat{p}(y) = \frac{N_y}{N} \quad$$
    Here, $N_y$ is the number of examples belonging to class $y$, and $N$ is the total number of examples.

*   **Estimated Data Distribution (Denominator)**:
    $$\hat{p}(x) = \sum_{i=1}^{C} \hat{p}(x|y_i)\hat{p}(y_i) \quad$$
    The unconditional probability $\hat{p}(x)$ is estimated by summing the product of the estimated class-conditional probability $\hat{p}(x|y_i)$ and the estimated class prior $\hat{p}(y_i)$ over all $C$ classes.

---

### 2. Parametric Classification: Gaussian Distribution

A very common model assumed for the class conditional probability, $p(x|y)$, is the Gaussian distribution, which models a 'blob'-like distribution in a $p$-dimensional feature space.

#### Gaussian Probability Density Function (PDF)
The general formula for a $p$-dimensional Gaussian distribution, $p(\mathbf{x})$, is:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}\det(\mathbf{\Sigma})^{1/2}} \exp \left(-\frac{1}{2}(\mathbf{x}-\mathbf{\mu})^T \mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{\mu})\right) \quad$$

Key parameters within this formula are:

*   **$\mathbf{\mu}$**: The **mean vector** of the distribution. This is often represented as a column vector.
*   **$\mathbf{\Sigma}$**: The **covariance matrix**, which defines the (elliptical) shape of the distribution.
*   **$\det(\mathbf{\Sigma})$**: The determinant of the covariance matrix.
*   **$(\mathbf{x}-\mathbf{\mu})^T \mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{\mu})$**: This term is known as the Mahalanobis distance squared.

#### Gaussian Parameter Estimation
To use the Gaussian distribution for classification (Plug-in Gaussian Distribution), the parameters ($\mathbf{\mu}$ and $\mathbf{\Sigma}$) must be estimated from the training data, typically using maximum likelihood estimators.

*   **Estimated Mean ($\mathbf{\hat{\mu}}$)**:
    $$\mathbf{\hat{\mu}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i \quad$$
    This is the sample mean vector calculated over $N$ training examples $\{\mathbf{x}_1, ..., \mathbf{x}_N\}$.

*   **Estimated Covariance Matrix ($\mathbf{\hat{\Sigma}}$)**:
    $$\mathbf{\hat{\Sigma}} = \frac{1}{N} \sum_{i=1}^{N} (\mathbf{x}_i - \mathbf{\hat{\mu}})(\mathbf{x}_i - \mathbf{\hat{\mu}})^T \quad$$
    This formula calculates the covariance matrix using the outer product $(\mathbf{x}_i - \mathbf{\hat{\mu}})(\mathbf{x}_i - \mathbf{\hat{\mu}})^T$. Note that when estimating the covariance matrix for a specific class $k$, the notation may use $n$ or $n_k$ for the number of examples in that class.

*   **Estimated Class-Conditional PDF**:
    $$\hat{p}(\mathbf{x}|y) = \frac{1}{\sqrt{(2\pi)^p\det(\mathbf{\hat{\Sigma}}_y)}} \exp \left(-\frac{1}{2}(\mathbf{x}-\mathbf{\hat{\mu}}_y)^T \mathbf{\hat{\Sigma}}_y^{-1}(\mathbf{x}-\mathbf{\hat{\mu}}_y)\right) \quad$$
    This uses the estimated mean $\mathbf{\hat{\mu}}_y$ and covariance matrix $\mathbf{\hat{\Sigma}}_y$ for class $y$.

---

### 3. Discriminant Functions (Quadratic and Linear Classifiers)

Discriminant functions are derived from the logarithm of the posterior probability $\hat{p}(y|x)$. The $p(x)$ term in Bayes' rule is independent of the class and can be dropped for classification purposes.

#### General Classification Rule
An object $\mathbf{x}$ is assigned to class $y_i$ when its discriminant function value is the highest among all classes $j$:
$$\text{Assign } \mathbf{x} \text{ to class } y_i \text{ when } g_i(\mathbf{x}) > g_j(\mathbf{x}) \text{ for all } i \ne j \quad$$
where $g_i(\mathbf{x})$ is defined as the log posterior probability (or an equivalent proportional function):
$$g_i(\mathbf{x}) = \log(\hat{p}(y_i|\mathbf{x})) \quad$$

#### Quadratic Discriminant Analysis (QDA)
QDA is based on the assumption that each class $y_i$ follows a Gaussian distribution with its own mean ($\mathbf{\mu}_i$) and its own unique covariance matrix ($\mathbf{\Sigma}_i$).

The discriminant function $g_i(\mathbf{x})$ is derived from the logarithm of the estimated class posterior probability (after dropping the class-independent term $\log p(\mathbf{x})$):

$$g_i(\mathbf{x}) = -\frac{1}{2} \log(\det\mathbf{\Sigma}_i) - \frac{1}{2} (\mathbf{x}-\mathbf{\mu}_i)^T \mathbf{\Sigma}_i^{-1} (\mathbf{x}-\mathbf{\mu}_i) + \log p(y_i) \quad$$

For the two-class case, the discriminant function $f(\mathbf{x})$ is defined as the difference between the log posteriors:
$$f(\mathbf{x}) = \log p(y_1|\mathbf{x}) - \log p(y_2|\mathbf{x}) \quad$$
An object is assigned to class $y_1$ if $f(\mathbf{x}) > 0$ and to class $y_2$ if $f(\mathbf{x}) < 0$.

The general form of the **quadratic discriminant** function is given by:
$$f(\mathbf{x}) = \mathbf{x}^T\mathbf{W}\mathbf{x} + \mathbf{w}^T\mathbf{x} + w_0 \quad$$
where the matrix $\mathbf{W}$, vector $\mathbf{w}$, and scalar $w_0$ depend on the estimated means and covariances of the two classes:
*   $$\mathbf{W} = \frac{1}{2} \left(\mathbf{\hat{\Sigma}}_2^{-1} - \mathbf{\hat{\Sigma}}_1^{-1}\right) \quad$$
*   $$\mathbf{w} = \mathbf{\hat{\mu}}_1^T \mathbf{\hat{\Sigma}}_1^{-1} - \mathbf{\hat{\mu}}_2^T \mathbf{\hat{\Sigma}}_2^{-1} \quad$$
*   $$w_0 = -\frac{1}{2} \log \det\mathbf{\hat{\Sigma}}_1 - \frac{1}{2} \mathbf{\hat{\mu}}_1^T \mathbf{\hat{\Sigma}}_1^{-1} \mathbf{\hat{\mu}}_1 + \log p(y_1) + \frac{1}{2} \log \det\mathbf{\hat{\Sigma}}_2 + \frac{1}{2} \mathbf{\hat{\mu}}_2^T \mathbf{\hat{\Sigma}}_2^{-1} \mathbf{\hat{\mu}}_2 - \log p(y_2) \quad$$

#### Linear Discriminant Analysis (LDA) and Nearest Mean Classifier
Linear Discriminant Analysis (LDA) is a simplification of QDA where all class covariance matrices are assumed to be equal ($\mathbf{\Sigma}_1 = \mathbf{\Sigma}_2 = \mathbf{\hat{\Sigma}}$). When this assumption holds, the discriminant function simplifies to a linear form:

$$f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + w_0 \quad$$
where:
*   $$\mathbf{w} = \mathbf{\hat{\Sigma}}^{-1}(\mathbf{\hat{\mu}}_2 - \mathbf{\hat{\mu}}_1) \quad$$
*   $$w_0 = \frac{1}{2} \mathbf{\hat{\mu}}_2^T \mathbf{\hat{\Sigma}}^{-1}\mathbf{\hat{\mu}}_2 - \frac{1}{2} \mathbf{\hat{\mu}}_1^T \mathbf{\hat{\Sigma}}^{-1}\mathbf{\hat{\mu}}_1 + \log \frac{p(y_1)}{p(y_2)} \quad$$

The **Nearest Mean Classifier** is a further simplification where the covariance matrix is assumed to be isotropic (all features have the same variance, and are uncorrelated): $\mathbf{\hat{\Sigma}} = \sigma^2\mathbf{I}$.

The class discriminant function for the nearest mean classifier is:
$$g_i(\mathbf{x}) = -\frac{1}{\sigma^2} \left( \frac{1}{2} \mathbf{\hat{\mu}}_i^T \mathbf{\hat{\mu}}_i - \mathbf{\hat{\mu}}_i^T \mathbf{x} \right) + \log(p(y_i)) \quad$$
This classifier essentially measures the distance to the mean of each class. The linear discriminant function $f(\mathbf{x})$ simplifies to:
*   $$\mathbf{w} = \mathbf{\hat{\mu}}_1 - \mathbf{\hat{\mu}}_2 \quad$$
*   $$w_0 = \frac{1}{2} \mathbf{\hat{\mu}}_2^T \mathbf{\hat{\mu}}_2 - \frac{1}{2} \mathbf{\hat{\mu}}_1^T \mathbf{\hat{\mu}}_1 + \sigma^2 \log \frac{p(y_1)}{p(y_2)} \quad$$

---

### 4. Non-Parametric Density Estimation

Non-parametric methods estimate the probability density directly from the data without assuming a fixed functional form (like Gaussian).

#### Histogram Method
For a single feature $\mathbf{x}$, the density estimate $\hat{p}(\mathbf{x})$ in a histogram bin of width $h$ is:
$$\hat{p}(\mathbf{x}) = \hat{p}(\mathbf{\hat{x}}) \approx \frac{1}{h} \frac{k_N}{N}, \quad |\mathbf{x}-\mathbf{\hat{x}}| \le h/2 \quad$$
where $k_N$ is the count (number of objects) in that region, and $N$ is the total number of objects.

#### Parzen Density Estimation
Parzen estimation uses kernel functions to estimate density.
*   **Uniform Kernel**: Defines the cell shape $K(\mathbf{r}, h)$ with volume $V$:
    $$K(\mathbf{r}, h) = \begin{cases} 0 & \text{if } |\mathbf{r}| > h \\ 1/V & \text{if } |\mathbf{r}| \le h \end{cases} \quad$$
*   **Parzen Density Estimate**: For a test object $\mathbf{z}$:
    $$\hat{p}(\mathbf{z}|h) = \frac{1}{n} \sum_{i=1}^{n} K(|\mathbf{z}-\mathbf{x}_i|, h) \quad$$
    This sums the contributions of kernels centered at each of the $n$ training data points $\mathbf{x}_i$. The parameter $h$ (the kernel width) is crucial for controlling smoothness.

*   **Parzen Width Heuristic**: A complex heuristic is provided for optimizing the parameter $h$:
    $$h = \sigma \left(\frac{4}{p+2}\right)^{1/(p+4)} n^{-1/(p+4)} \quad$$
    Where $\sigma^2$ is an average variance estimate across features.

#### K-Nearest Neighbor (k-NN) Density Estimate
k-NN fixes the number of neighbors ($k$) rather than the volume.
*   **Density Estimate**:
    $$\hat{p}(\mathbf{x}) = \frac{k}{n V_k} \quad$$
    Here, $k$ is the number of neighbors (e.g., $k=3$), $n$ is the total number of training examples, and $V_k$ is the volume of the sphere centered at $\mathbf{x}$ necessary to enclose $k$ objects (the distance to the k-th nearest neighbor defines the radius).

*   **Class Conditional Density**: For class $y_m$:
    $$\hat{p}(\mathbf{x}|y_m) = \frac{k_m}{n_m V_k} \quad$$
    $k_m$ is the count of neighbors belonging to class $y_m$, and $n_m$ is the total number of training examples of class $y_m$.

*   **Classification Rule Simplification**: By applying Bayes' rule using the k-NN density estimates (and the class prior $\hat{p}(y_m) = n_m/n$), the classification rule simplifies:
    $$\hat{p}(\mathbf{x}|y_m)\hat{p}(y_m) > \hat{p}(\mathbf{x}|y_i)\hat{p}(y_i) \implies k_m > k_i \quad$$
    This means an object is assigned to the class that has the most representatives among the $k$ nearest neighbors.

*   **Limiting Error**: The error ($\epsilon$) of the k-NN classifier when $k$ approaches the total number of samples $N$ is limited by the minimum class prior:
    $$\lim_{k \to N} \epsilon_{k\text{-NN}} = \min(p(y_1), p(y_2)) \quad$$