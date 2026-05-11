# Recursive Alternating Amplitude Embedding (RAAE)

Official Python implementation of **Recursive Alternating Amplitude Embedding (RAAE)**, a hybrid quantum circuit architecture for balanced depth and width.

## 📖 Introduction
Quantum state preparation is a fundamental step in quantum information processing. RAAE is a novel framework that addresses the inherent trade-offs between circuit width (qubit count) and circuit depth.

Unlike traditional methods that are either qubit-efficient but very deep (Möttönen) or shallow but qubit-heavy (Divide-and-Conquer), RAAE recursively alternates between these two strategies. This results in a balanced architecture that scales more efficiently for high-dimensional classical data embedding.

## ✨ Key Features
* **Balanced Resource Scaling**: Optimized trade-off between parallel execution and qubit conservation.
* **Exponential Qubit Savings**: Reduces width scaling from $O(n 2^{n/2})$ to $O(2^{n/2})$ compared to hybrid methods like BDSP.
* **Depth Efficiency**: Achieves approximately $1/3$ to $2/3$ depth reduction compared to the standard Möttönen algorithm for $R_Y$ and CNOT gates.
* **CNOT Reduction**: Effectively removes $O(2^{n/2})$ CNOT gates compared to purely sequential methods.

## 🛠 Installation
The library requires `numpy`, `matplotlib`, and `qiskit`. Install the dependencies using:

```bash
pip install numpy matplotlib qiskit qiskit-aer
```

## 🚀 Usage Example

The following example demonstrates how to encode a classical vector into a 4-qubit quantum state using the RAAE library.

---

### 📦 Initialization

```python
import numpy as np
from qiskit_aer import AerSimulator
import raae_state_preparation as raae_sp

# Prepare normalized classical data
num_qubits = 4
data = np.random.rand(2**num_qubits)
target_vector = raae_sp.normalize(data)

# Initialize circuit builder
builder = raae_sp.RAAECircuitBuilder(2**num_qubits)

# Create parameterized circuit with measurement
qc = builder.get_circuit_with_measurement()

# Create state converter
converter = raae_sp.StateConverter()

# Convert state vector into circuit parameters
params = converter.convert_to_params(target_vector)

# Bind parameters into the circuit
bound_qc = qc.assign_parameters(params)
```

---

### ▶️ Run

```python
backend = AerSimulator()

result = backend.run(
    bound_qc,
    shots=16384
).result()
```

---

### 📈 Get Result

```python
counts = result.get_counts()

# Convert counts into probabilities
measured_probs = converter.to_probabilities(counts)

print("Target Probabilities (first 5):")
print((target_vector**2)[:5])

print("\nMeasured Probabilities (first 5):")
print(measured_probs[:5])
```

---

## 📖 Citation

If you use RAAE in your research, please cite our paper.
