


<div id="header" align="center">
  <img src="https://media.giphy.com/media/M9gbBd9nbDrOTu1Mqx/giphy.gif" width="100"/>
</div>

### Digit count array:
[1309, 1576, 948, 1273, 1186, 1301, 1222, 745, 1006, 1434]



Sum: **12000**, files: **12000**

#### Чтобы установить зависимости на другом компьютере:

pip install -r requirements.txt
---

## 1. Key Features

- Convolutional Neural Network trained on MNIST  
- All randomness eliminated: full determinism  
- Saved model (`mnist_model.h5`) ensures identical output on every run  
- CPU-only execution  
- No non-deterministic TensorFlow kernels  
- Output is 100% reproducible across machines and operating systems  

---

## 2. Deterministic Execution Guarantee

This solution guarantees identical results on every run due to:

1. `PYTHONHASHSEED`, `numpy`, `random` seeds fixed to 42  
2. `TF_DETERMINISTIC_OPS=1`  
3. `TF_ENABLE_ONEDNN_OPTS=0`  
4. XLA disabled  
5. Model batch order is not shuffled (`shuffle=False`)  
6. Model is trained **once** and saved  
7. All subsequent runs reuse the saved model  
