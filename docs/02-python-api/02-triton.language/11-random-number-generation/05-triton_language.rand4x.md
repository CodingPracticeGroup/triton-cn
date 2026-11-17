---
title: triton_language.rand4x
---

```python
triton.language.rand4x(seed, offsets, n_rounds=N_ROUNDS_DEFAULT)
```

给定 1 个 `seed` 标量和 1 个 `offsets` 块，返回 4 个在 $$U(0, 1)$$ 中的随机 `float32` 块。

**参数：**

* **seed** - 用于生成随机数的种子。
* **offsets** - 用于生成随机数的偏移量。
