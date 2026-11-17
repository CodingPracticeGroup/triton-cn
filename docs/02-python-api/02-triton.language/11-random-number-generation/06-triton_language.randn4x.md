---
title: triton_language.randn4x
---

```python
triton.language.randn4x(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT)
```

给定 1 个 `seed` 标量和 1 个 `offset` 块，返回 4 个在 $$\mathcal{N}(0, 1)$$ 中的随机 `float32` 块。

**参数：**

* **seed** - 用于生成随机数的种子。
* **offsets** - 用于生成随机数的偏移量。
