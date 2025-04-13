# 🔄 NICE: Non-linear Independent Components Estimation

A PyTorch implementation of the NICE (Non-linear Independent Components Estimation) generative model, originally proposed by Dinh et al. (2014). This project was completed as part of the Generative Models course in an MSc Data Science program.

---

## 🧱 Model Architecture

Below is a schematic illustration of the NICE model structure with alternating additive coupling layers and a final scaling layer:

![NICE Model Architecture](https://miro.medium.com/v2/resize:fit:1400/0*NH5VVAcdtUkQKTkF.png)

---

## 📦 Project Structure

```
.
├── data/             # Stores downloaded datasets (e.g., MNIST)
├── models/           # Placeholder for saving trained models
├── samples/          # Generated image samples
├── nice.py           # Core NICE model and coupling layer definitions
├── train.py          # Training and evaluation script
├── req.txt           # Python dependencies
├── report.pdf        # Report file (e.g., for course submission)
```

---

## 🔍 About the NICE Model

NICE is a type of normalizing flow model that learns an invertible transformation from a simple prior (e.g., logistic or Gaussian) to a complex data distribution. This implementation supports:

- **Additive Coupling Layers**
- **Affine Coupling Layers**
- **Scaling Layers**
- **Sampling and Density Estimation**
- **MNIST and Fashion-MNIST support**

---

## 🧪 Example Usage

### Training

```bash
python train.py \
  --dataset mnist \
  --prior logistic \
  --batch_size 128 \
  --epochs 50 \
  --coupling-type additive \
  --coupling 4 \
  --mid-dim 1000 \
  --hidden 5 \
  --lr 0.001
```

### Options

| Argument         | Description                                       | Default     |
|------------------|---------------------------------------------------|-------------|
| `--dataset`       | Dataset to train on (`mnist` or `fashion-mnist`) | `mnist`     |
| `--prior`         | Latent distribution (`logistic` or `gaussian`)   | `logistic`  |
| `--batch_size`    | Mini-batch size                                  | `128`       |
| `--epochs`        | Number of training epochs                        | `50`        |
| `--coupling-type` | Type of coupling layer (`additive`, `adaptive`)  | `additive`  |
| `--coupling`      | Number of coupling layers                        | `4`         |
| `--mid-dim`       | Width of hidden layers                           | `1000`      |
| `--hidden`        | Number of hidden layers in each coupling net     | `5`         |
| `--lr`            | Learning rate                                    | `1e-3`      |

---

## 📊 Sample Outputs

- 📁 Saved samples appear in `./samples/`
- 📈 Loss plots are saved automatically as `.png` files (train/test curves)

---

## 🧠 Requirements

Install the required packages via:

```bash
pip install -r req.txt
```

---

## 📚 References

- Laurent Dinh, David Krueger, and Yoshua Bengio. [NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516), 2014.

---

## 👨‍🎓 Author

This project was completed as part of the *Generative Models* course in the MSc in Data Science program.
