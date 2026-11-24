# Ford Ranger Models Price Prediction

### Trained Model Parameters (θ)

| Feature Scaling | Bound Type | θ₀ (Intercept) | θ₁ (Slope) |
| :--- | :--- | :---: | :---: |
| **Unscaled** | Minimum | `0.0144` (`0.014409599969484708`) | `9.7351` (`9.735110701960915`) |
| **Unscaled** | Maximum | `0.0180` (`0.01797665426659515`) | `16.7313` (`16.73130004807617`) |
| **Scaled** | Minimum | `0.0152` (`0.015205321252072008`) | `0.7787` (`0.7786728916402945`) |
| **Scaled** | Maximum | `0.0120` (`0.01202170683764288`) | `0.5212` (`0.5211963180600709`) |

### Model Analysis

Comparing the unscaled θ values to the scaled θ values reveals a clear difference—especially in θ₁, which heavily affects the slope of the minimum and maximum price predictions. 

* **Unscaled Model:** The model performed poorly because all predicted prices cluster around the same amount (for example, minimum price predictions stay flat around \$19k), resulting in a nearly flat horizontal line on the graph rather than reflecting actual price trends across model years.
* **Scaled Model:** Feature scaling produces a clear positive slope on the graph and realistic price variation across years (such as minimum predicted prices ranging from \$19k to \$23k).
* **2026 Next-Generation Ranger Prediction:** The model predicts a price range of **\$27,593.58 – \$39,872.41**. I would not rely on this estimate to buy a 2026 Ranger because the model underpredicts actual market prices. Prices jumped by roughly \$10,000 between 2023 and 2024 due to inflation, and the large volume of older, lower historical prices pulls the regression line downward.
* **Future Improvement:** Filtering out pre-2000s vehicle data would remove older low-price outliers and shift the predictions closer to modern market pricing.

---

### Mathematical Formulation

**Cost Function**

```math
J(\theta) = \frac{1}{m}\sum_{i=1}^{m}\left((e^{(i)})^2 + 0.1(e^{(i)})^4\right)
```

**Error Definition**

```math
e^{(i)} = h_{\theta}(x^{(i)}) - y^{(i)}
```

**Gradient with Respect to θ₀**

```math
\frac{\partial J}{\partial \theta_{0}} = \frac{1}{m}\sum_{i=1}^{m}\left(2e^{(i)} + 0.4(e^{(i)})^3\right)
```

**Gradient with Respect to θ₁**

```math
\frac{\partial J}{\partial \theta_{1}} = \frac{1}{m}\sum_{i=1}^{m}\left(2e^{(i)} + 0.4(e^{(i)})^3\right)x^{(i)}
```
