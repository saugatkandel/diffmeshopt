# Analysis of Central Limit Theorem (CLT) Application in Template Shape Loss

> **Note:** I don't clearly understand the CLT application here, and I need to think about this further.

## 1. The Theoretical Argument (Why CLT suggests L2)

The Central Limit Theorem states that the distribution of the **sample mean** of independent random variables approaches a Gaussian distribution as the sample size $N$ increases, regardless of the shape of the original distribution (provided it has finite variance).

In `TemplateShapeLoss`, the operation performed is:
```python
mean_profile = profiles.mean(dim=0)
```

For any specific pixel index $x$ along the profile (e.g., the center pixel), we have $N$ samples $I_1(x), I_2(x), ..., I_N(x)$.
According to the CLT, the error of the calculated mean $\bar{I}(x)$ relative to the **true population mean** $\mu(x)$ is normally distributed:
$$ \bar{I}(x) \sim \mathcal{N}\left(\mu(x), \frac{\sigma^2}{N}\right) $$

Because the "noise" (sampling error) on this mean profile is Gaussian, the Maximum Likelihood Estimator (MLE) for fitting a model $T(x)$ to this data is the **Least Squares (L2)** estimate. This is the standard statistical justification for using MSE.

## 2. The Flaw: The "Mixture" Problem

The CLT guarantees that the `mean_profile` converges very precisely to the **Population Mean**. The problem is that the population contains both "Signal" (intact membrane) and "Outliers" (broken membrane/background).

Let $\alpha$ be the fraction of broken membranes. The population mean at pixel $x$ is:
$$ \mu(x) = (1-\alpha) \cdot \text{Signal}(x) + \alpha \cdot \text{Background}(x) $$

*   **Signal(x)**: The Gaussian peak representing the membrane.
*   **Background(x)**: A flat line or noise floor.

The resulting $\mu(x)$ is a **mixture**: it looks like the signal, but "squashed" and sitting on a lifted baseline (the background contribution). It effectively has "fat tails."

**The CLT works perfectly:** it ensures a very stable, low-noise estimate of this *corrupted* mixture shape. It does **not** remove the corruption.

## 3. Why L2 Maximizes Sigma (The Outlier Effect)

When fitting a clean Gaussian template $T(x; \sigma)$ to this corrupted mean profile $\mu(x)$ using L2 loss:

$$ L_{MSE} = \sum (T(x; \sigma) - \mu(x))^2 $$

1.  **The Tails**: In the tail regions (far from center), the Template $T$ wants to be 0. However, the Data $\mu$ is $>0$ (due to the background mixture).
2.  **Quadratic Penalty**: L2 squares this error. A constant background offset creates a significant penalty.
3.  **The "Fix"**: To minimize this squared error, the optimizer widens the template (increases $\sigma$). By making the Gaussian wider, the template's tails lift up, covering the background noise.

**Result**: The optimizer sacrifices the fit at the peak (making it too fat) to reduce the massive penalty coming from the tails.

## 4. Why L1 is More Robust

$$ L_{MAE} = \sum |T(x; \sigma) - \mu(x)| $$

L1 penalizes the error linearly. The "cost" of the background offset in the tails is constant regardless of how wide the template is (mostly). The optimizer gains more by matching the sharp shape of the central peak than by trying to cover the wide tails.

## 5. The True Robust Solution: Median

To strictly rely on the "Signal" and ignore the "Broken" regions, one should not use the Mean (which invokes CLT on the mixture). One should use the **Median**.

If the membrane is intact in $>50\%$ of the samples, the **Median Profile** will track the Signal and completely ignore the Background.

## 6. Resolution: Beyond CLT (Confidence & Robustness)

### L1 vs L2
Given the "Mixture Problem" (Signal + Background), **L1 (MAE)** is the superior choice for the Shape Loss. It is robust to the "fat tails" caused by the background intensity, whereas L2 will artificially widen the template to suppress the quadratic error in the tails.

### The "Median" Limitation
While the Median is ideal for extracting a global canonical profile, it is ill-suited for **Per-Point** or **Spatially Varying** models. A per-point model fits a unique distribution to local data; there is no "population" to take the median of (unless we aggregate neighbors, which introduces blurring).

### Handling Broken Membranes: The Pragmatic Approach

Instead of implementing complex uncertainty estimation (which risks optimization instability), we can achieve the project goals with simpler tools:

1.  **Robustness via L1**: Use **L1 Loss (MAE)** for the Shape Loss. This prevents the template width ($\sigma$) from exploding to cover the background noise (fat tails), ensuring the template fits the central membrane peak even in "mixture" regions.
2.  **Confidence via Correlation**: The optimization objective is maximizing Cross-Correlation.
    *   **Post-Hoc Analysis**: Simply visualize the per-point **Correlation Score**.
    *   **Interpretation**: Regions with low correlation scores correspond to broken membranes or protein distortions. No extra learned parameters are required.