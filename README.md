Question Bank — Full Code + Line-by-Line Explanation
Q1 — Single Perceptron Regression (House Price Prediction)

python
import numpy as np
import matplotlib.pyplot as plt

X = np.array([500, 800, 1000, 1200, 1500], dtype=float)
Y = np.array([150, 220, 300, 360, 450], dtype=float)
X_norm = X / X.max()
Y_norm = Y / Y.max()

w = 0.0
b = 0.0
lr = 0.01
epochs = 1000
losses = []

for epoch in range(epochs):
    y_pred = w * X_norm + b
    loss = np.mean((y_pred - Y_norm) ** 2)
    losses.append(loss)
    dw = np.mean((y_pred - Y_norm) * X_norm)
    db = np.mean(y_pred - Y_norm)
    w -= lr * dw
    b -= lr * db

x_new = 1100 / X.max()
predicted = (w * x_new + b) * Y.max()
print(f"Predicted price for 1100 sqft: {predicted:.2f} thousand")

plt.subplot(1,2,1)
plt.plot(losses)
plt.title("Loss Reduction")

plt.subplot(1,2,2)
plt.scatter(X, Y, color='blue', label='Actual')
x_line = np.linspace(400, 1600, 100)
y_line = (w * (x_line / X.max()) + b) * Y.max()
plt.plot(x_line, y_line, color='red', label='Regression Line')
plt.legend()
plt.show()
Line by line:

import numpy as np — loads NumPy for math operations on arrays. import matplotlib.pyplot as plt — loads plotting library. X = np.array([500,800,1000,1200,1500], dtype=float) — house areas in sqft. Y = np.array([150,220,300,360,450], dtype=float) — house prices in thousands. X_norm = X / X.max() — normalizes X to range [0,1] by dividing by max (1500). Normalization prevents large input values from causing unstable gradients. Y_norm = Y / Y.max() — normalizes Y to [0,1] the same way. w = 0.0 — weight initialized to zero. This is the slope of the line. b = 0.0 — bias initialized to zero. This is the y-intercept. lr = 0.01 — learning rate controls how big each update step is. epochs = 1000 — number of times the full dataset is passed through. losses = [] — empty list to track loss history for plotting. y_pred = w * X_norm + b — perceptron prediction: weighted input plus bias. This is the single neuron's forward pass. loss = np.mean((y_pred - Y_norm) ** 2) — Mean Squared Error: average of squared differences between predicted and actual. losses.append(loss) — saves current loss for the plot. dw = np.mean((y_pred - Y_norm) * X_norm) — gradient of loss w.r.t. weight (chain rule: d(MSE)/dw). db = np.mean(y_pred - Y_norm) — gradient of loss w.r.t. bias. w -= lr * dw — update weight by stepping in the opposite direction of gradient. b -= lr * db — same for bias. x_new = 1100 / X.max() — normalize the new input (1100 sqft) the same way training data was normalized. predicted = (w * x_new + b) * Y.max() — predict then de-normalize back to actual price scale.

Q2 — MLP: Student Pass/Fail

python
import numpy as np

X = np.array([[2,50],[4,60],[6,70],[8,80]], dtype=float)
Y = np.array([[0],[0],[1],[1]], dtype=float)
X = X / np.array([10, 100])

np.random.seed(42)
W1 = np.random.randn(2, 3) * 0.1
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.1
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

lr = 0.1
for epoch in range(50):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    loss = np.mean((Y - a2) ** 2)
    d_a2 = -(Y - a2)
    d_z2 = d_a2 * sigmoid_deriv(a2)
    d_W2 = a1.T @ d_z2
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)
    d_a1 = d_z2 @ W2.T
    d_z1 = d_a1 * sigmoid_deriv(a1)
    d_W1 = X.T @ d_z1
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)
    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 | Loss: {loss:.4f}")

predictions = (a2 > 0.5).astype(int)
accuracy = np.mean(predictions == Y) * 100
print(f"Accuracy: {accuracy:.1f}%")

test = np.array([[5, 65]]) / np.array([10, 100])
z1t = sigmoid(test @ W1 + b1)
a2t = sigmoid(z1t @ W2 + b2)
print(f"Prediction for (5hrs, 65%): {'Pass' if a2t[0][0]>0.5 else 'Fail'}")
Line by line:

X = np.array([[2,50],[4,60],[6,70],[8,80]]) — each row is one student: [study hours, attendance %]. Y = np.array([[0],[0],[1],[1]]) — 0 = Fail, 1 = Pass. X = X / np.array([10, 100]) — normalizes column 1 by 10 (max hours) and column 2 by 100 (max %). Keeps both features in [0,1] range. W1 = np.random.randn(2, 3) * 0.1 — weight matrix from input layer (2 neurons) to hidden layer (3 neurons). Multiplied by 0.1 to keep initial weights small. b1 = np.zeros((1, 3)) — one bias per hidden neuron, all starting at 0. W2 = np.random.randn(3, 1) * 0.1 — weight matrix from hidden layer (3) to output (1). b2 = np.zeros((1, 1)) — single output bias. def sigmoid(x): return 1/(1+np.exp(-x)) — squashes any value into (0,1). Used as activation for both layers since output is binary. def sigmoid_deriv(x): return x*(1-x) — derivative of sigmoid. Used in backprop. Note: x here is the already-activated output, not raw z. z1 = X @ W1 + b1 — hidden layer pre-activation: matrix multiply input with W1 then add bias. a1 = sigmoid(z1) — apply sigmoid to get hidden layer output. z2 = a1 @ W2 + b2 — output layer pre-activation. a2 = sigmoid(z2) — final prediction between 0 and 1. loss = np.mean((Y - a2)**2) — MSE loss. d_a2 = -(Y - a2) — derivative of MSE loss w.r.t. a2. The negative sign comes from d/da2 of (Y-a2)². d_z2 = d_a2 * sigmoid_deriv(a2) — chain rule: multiply by sigmoid derivative to get gradient at z2. d_W2 = a1.T @ d_z2 — gradient of loss w.r.t. W2. d_b2 = np.sum(d_z2, axis=0, keepdims=True) — sum gradients across all samples for bias. d_a1 = d_z2 @ W2.T — propagate gradient back through W2 to hidden layer. d_z1 = d_a1 * sigmoid_deriv(a1) — apply sigmoid derivative at hidden layer. d_W1 = X.T @ d_z1 — gradient of loss w.r.t. W1. W2 -= lr * d_W2 — update W2 in direction that reduces loss. predictions = (a2 > 0.5).astype(int) — threshold at 0.5 to get binary 0/1 class. test = np.array([[5,65]]) / np.array([10,100]) — normalize the new test input the same way.

Q3 — Transformer Components From Scratch

python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V, weights

def multi_head_attention(X, num_heads=2):
    d_model = X.shape[-1]
    d_head = d_model // num_heads
    outputs = []
    for _ in range(num_heads):
        Wq = np.random.randn(d_model, d_head) * 0.1
        Wk = np.random.randn(d_model, d_head) * 0.1
        Wv = np.random.randn(d_model, d_head) * 0.1
        out, _ = scaled_dot_product_attention(X @ Wq, X @ Wk, X @ Wv)
        outputs.append(out)
    return np.concatenate(outputs, axis=-1)

def feed_forward(X, d_ff=16):
    d_model = X.shape[-1]
    W1 = np.random.randn(d_model, d_ff) * 0.1
    W2 = np.random.randn(d_ff, d_model) * 0.1
    return np.maximum(0, X @ W1) @ W2

def layer_norm(X, eps=1e-6):
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    return (X - mean) / (std + eps)

def transformer_encoder(X):
    X = X + positional_encoding(X.shape[0], X.shape[1])
    attn_out = multi_head_attention(X)
    X = layer_norm(X + attn_out)
    ff_out = feed_forward(X)
    X = layer_norm(X + ff_out)
    return X

np.random.seed(42)
X = np.array([[1,0,1],[0,1,1],[1,1,0]], dtype=float)
output = transformer_encoder(X)
print("Output:\n", output)

Wo = np.random.randn(X.shape[1], X.shape[1]) * 0.1
next_vec = output[-1] @ Wo
print("Predicted next vector:", next_vec)
Line by line:

e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) — subtract max before exponentiation to prevent overflow. This is numerically stable softmax. return e_x / e_x.sum(axis=-1, keepdims=True) — divide each value by the sum of the row so all values sum to 1 (probability distribution).

PE = np.zeros((seq_len, d_model)) — create empty positional encoding matrix, one row per position in sequence. PE[pos, i] = np.sin(pos / (10000 ** (i/d_model))) — even dimensions get sine encoding. Encodes absolute position in a continuous way. PE[pos, i+1] = np.cos(...) — odd dimensions get cosine. Sine+cosine pair lets the model distinguish positions.

d_k = Q.shape[-1] — dimension of the query/key vectors. scores = Q @ K.T / np.sqrt(d_k) — dot product of Q and K gives similarity scores. Divide by √d_k to prevent scores from becoming too large (which would make softmax near-zero gradients). weights = softmax(scores) — convert scores to attention probabilities (which tokens to focus on). return weights @ V — weighted sum of value vectors — the output is a blend of values weighted by how relevant each token is.

d_head = d_model // num_heads — split model dimension equally across heads. for _ in range(num_heads): — each head learns different attention patterns independently. Wq, Wk, Wv — each head has its own projection matrices (randomly initialized here). np.concatenate(outputs, axis=-1) — join all heads' outputs back together to restore original d_model dimension.

np.maximum(0, X @ W1) @ W2 — feed-forward: first linear layer + ReLU activation, then second linear layer. Adds non-linearity and processes each token independently.

(X - mean) / (std + eps) — Layer Normalization: zero-mean, unit-variance per token. Stabilizes training. eps prevents division by zero.

X = X + positional_encoding(...) — add position info to input embeddings (residual addition). X = layer_norm(X + attn_out) — add attention output to input (residual connection) then normalize. Residual connections prevent vanishing gradients. X = layer_norm(X + ff_out) — same residual + norm after feed-forward.

next_vec = output[-1] @ Wo — take last token's output vector, multiply by output weight Wo to predict next vector.

Q4 — MLP for XOR

python
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
Y = np.array([[0],[1],[1],[0]], dtype=float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

for lr in [0.01, 0.1]:
    np.random.seed(0)
    W1 = np.random.randn(2, 4)
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1)
    b2 = np.zeros((1, 1))

    for epoch in range(10):
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        loss = np.mean((Y - a2)**2)
        d2 = -(Y - a2) * sigmoid_deriv(a2)
        dW2 = a1.T @ d2
        db2 = d2.sum(axis=0)
        d1 = (d2 @ W2.T) * sigmoid_deriv(a1)
        dW1 = X.T @ d1
        db1 = d1.sum(axis=0)
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    print(f"LR={lr} | Final Loss: {loss:.4f} | Predictions: {a2.round(3).flatten()}")
Line by line:

X = np.array([[0,0],[0,1],[1,0],[1,1]]) — all 4 XOR input combinations. Y = np.array([[0],[1],[1],[0]]) — XOR truth table: output 1 only when inputs differ. for lr in [0.01, 0.1]: — trains the same network twice with different learning rates to compare convergence. np.random.seed(0) — fixed seed inside the loop so both runs start from identical weights, making comparison fair. W1 = np.random.randn(2, 4) — 2 inputs → 4 hidden neurons. Larger hidden layer helps solve non-linear XOR. z1 = X @ W1 + b1 — all 4 inputs processed in one matrix operation (batch forward pass). a1 = sigmoid(z1) — hidden layer activations, shape (4 samples, 4 neurons). a2 = sigmoid(z2) — output prediction, shape (4 samples, 1). d2 = -(Y - a2) * sigmoid_deriv(a2) — output layer error gradient. d1 = (d2 @ W2.T) * sigmoid_deriv(a1) — backpropagate error from output to hidden layer through W2. dW1 = X.T @ d1 — gradient for W1: how much each weight contributed to the error. db1 = d1.sum(axis=0) — sum gradients across all 4 training samples to get one bias update.

With LR=0.01 the loss barely decreases in 10 epochs (too slow). With LR=0.1 it converges faster.

Q5 & Q6 — Transformer on Pixel Sequences

python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i+1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V

pixels = np.array([1, 0, 1, 1, 0], dtype=float)
d_model = 4
seq_len = len(pixels)

X = np.tile(pixels.reshape(-1, 1), (1, d_model))
X = X + positional_encoding(seq_len, d_model)

for lr in [0.001, 0.01]:
    np.random.seed(0)
    Wq = np.random.randn(d_model, d_model) * 0.1
    Wk = np.random.randn(d_model, d_model) * 0.1
    Wv = np.random.randn(d_model, d_model) * 0.1
    Wo = np.random.randn(d_model, 1) * 0.1

    for epoch in range(10):
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        attn_out = scaled_dot_product_attention(Q, K, V)
        pred = attn_out @ Wo
        target = pixels[-1]
        loss = np.mean((pred - target)**2)
        d_out = 2 * (pred - target) / len(pred)
        dWo = attn_out.T @ d_out
        Wo -= lr * dWo

    print(f"LR={lr} | Loss: {loss:.4f} | Predicted next pixel: {pred[-1][0]:.4f}")
Line by line:

pixels = np.array([1,0,1,1,0]) — a sequence of 5 pixel values treated like a time series. d_model = 4 — embedding dimension. Each pixel is represented as a 4-d vector. X = np.tile(pixels.reshape(-1,1), (1, d_model)) — reshape pixels to column, then tile 4 times: each pixel value repeated across all 4 dimensions to create a 5×4 matrix. X = X + positional_encoding(seq_len, d_model) — inject position information so the model knows pixel 1 comes before pixel 2, etc. Q = X @ Wq — project input X to Query space. Wq is a learned projection. K = X @ Wk — project to Key space. V = X @ Wv — project to Value space. attn_out = scaled_dot_product_attention(Q, K, V) — each pixel now attends to all other pixels weighted by similarity. pred = attn_out @ Wo — Wo is output projection (d_model → 1) to predict a single next pixel value. target = pixels[-1] — last pixel value is the target we're trying to predict. loss = np.mean((pred - target)**2) — MSE between all predicted values and the target. d_out = 2*(pred - target)/len(pred) — gradient of MSE. dWo = attn_out.T @ d_out — gradient w.r.t. output weight Wo. Wo -= lr * dWo — only Wo is updated here (simplified training). With LR=0.01 converges faster than 0.001.

Q7 — GAN Using Pixel Vectors

python
import numpy as np

real_data = np.array([[1,0,1,0],[0,1,0,1]], dtype=float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def sigmoid_deriv(x):
    return x * (1 - x)

for lr in [0.001, 0.01]:
    np.random.seed(42)
    G_W1 = np.random.randn(2, 4) * 0.1
    G_b1 = np.zeros((1, 4))
    D_W1 = np.random.randn(4, 4) * 0.1
    D_b1 = np.zeros((1, 4))
    D_W2 = np.random.randn(4, 1) * 0.1
    D_b2 = np.zeros((1, 1))

    for epoch in range(10):
        noise = np.random.randn(2, 2)
        fake = sigmoid(noise @ G_W1 + G_b1)

        real_h = sigmoid(real_data @ D_W1 + D_b1)
        real_out = sigmoid(real_h @ D_W2 + D_b2)
        fake_h = sigmoid(fake @ D_W1 + D_b1)
        fake_out = sigmoid(fake_h @ D_W2 + D_b2)

        d_loss = -np.mean(np.log(real_out+1e-8)) - np.mean(np.log(1-fake_out+1e-8))
        g_loss = -np.mean(np.log(fake_out + 1e-8))

        d_real = -(1 - real_out) * sigmoid_deriv(real_out)
        d_fake = fake_out * sigmoid_deriv(fake_out)
        dD_W2 = real_h.T @ d_real / 2 + fake_h.T @ d_fake / 2
        dD_b2 = (d_real + d_fake).mean(axis=0)
        d_real_h = d_real @ D_W2.T * sigmoid_deriv(real_h)
        d_fake_h = d_fake @ D_W2.T * sigmoid_deriv(fake_h)
        dD_W1 = real_data.T @ d_real_h / 2 + fake.T @ d_fake_h / 2
        dD_b1 = (d_real_h + d_fake_h).mean(axis=0)
        D_W2 -= lr * dD_W2
        D_b2 -= lr * dD_b2
        D_W1 -= lr * dD_W1
        D_b1 -= lr * dD_b1

        fake_h2 = sigmoid(fake @ D_W1 + D_b1)
        fake_out2 = sigmoid(fake_h2 @ D_W2 + D_b2)
        g_loss = -np.mean(np.log(fake_out2 + 1e-8))

        d_g_out = -(1/(fake_out2+1e-8)) * sigmoid_deriv(fake_out2)
        d_g_h = d_g_out @ D_W2.T * sigmoid_deriv(fake_h2)
        d_g_fake = d_g_h @ D_W1.T * sigmoid_deriv(fake)
        G_W1 -= lr * noise.T @ d_g_fake
        G_b1 -= lr * d_g_fake.mean(axis=0, keepdims=True)

    print(f"LR={lr} | D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f}")
    noise = np.random.randn(2, 2)
    generated = sigmoid(noise @ G_W1 + G_b1)
    print(f"Generated:\n{generated.round(3)}")
Line by line:

real_data = [[1,0,1,0],[0,1,0,1]] — two real binary patterns. Discriminator must learn these are "real." G_W1 = np.random.randn(2,4) — Generator weights: takes 2D noise → produces 4D fake pattern. D_W1, D_W2 — Discriminator is a 2-layer network: 4→4→1 (outputs single real/fake score).

Generator forward pass: noise = np.random.randn(2, 2) — random noise vector, the generator's input (2 samples of 2D noise). fake = sigmoid(noise @ G_W1 + G_b1) — Generator produces fake 4-element patterns from noise.

Discriminator forward pass: real_h = sigmoid(real_data @ D_W1 + D_b1) — hidden layer activations for real data. real_out = sigmoid(real_h @ D_W2 + D_b2) — D's confidence score for real data (should approach 1). fake_h / fake_out — same for fake data (D should output near 0 for fake).

Discriminator loss: d_loss = -mean(log(real_out)) - mean(log(1-fake_out)) — binary cross entropy. D wants to maximize log(D(real)) + log(1-D(fake)).

Discriminator backprop: d_real = -(1-real_out) * sigmoid_deriv(real_out) — gradient for real samples through D's output sigmoid. d_fake = fake_out * sigmoid_deriv(fake_out) — gradient for fake samples. dD_W2 = real_h.T @ d_real/2 + fake_h.T @ d_fake/2 — W2 gradient averaged across real and fake. d_fake_h = d_fake @ D_W2.T * sigmoid_deriv(fake_h) — propagate back through D's hidden layer. dD_W1 = real_data.T @ d_real_h/2 + fake.T @ d_fake_h/2 — W1 gradient.

Generator training (the critical fix): fake_h2 / fake_out2 — re-run fake through the UPDATED discriminator. g_loss = -mean(log(fake_out2)) — Generator wants D to output 1 for fake → minimize -log(D(G(z))). d_g_out = -(1/fake_out2) * sigmoid_deriv(fake_out2) — gradient of G loss w.r.t. D's output. d_g_h = d_g_out @ D_W2.T * sigmoid_deriv(fake_h2) — backprop through D's hidden layer. d_g_fake = d_g_h @ D_W1.T * sigmoid_deriv(fake) — backprop all the way to G's output. G_W1 -= lr * noise.T @ d_g_fake — finally update Generator weights using the gradient that flowed back from D.

Q8 — Self-Attention on Pixel Vectors

python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

pixels = np.array([[1, 0, 1, 0, 1]], dtype=float)
d_model = 5

np.random.seed(42)
Wq = np.random.randn(d_model, d_model) * 0.1
Wk = np.random.randn(d_model, d_model) * 0.1
Wv = np.random.randn(d_model, d_model) * 0.1

for lr in [0.001, 0.01]:
    Wv = np.random.randn(d_model, d_model) * 0.1
    for step in range(10):
        Q = pixels @ Wq
        K = pixels @ Wk
        V = pixels @ Wv
        scores = Q @ K.T / np.sqrt(d_model)
        weights = softmax(scores[0])
        attn_out = weights @ V
        loss = np.mean(attn_out**2)
        d_attn = 2 * attn_out / attn_out.size
        Wv -= lr * pixels.T @ (weights.reshape(1,-1) * d_attn.reshape(1,-1))

    print(f"LR={lr} | Attention weights: {weights.round(4)}")
    print(f"Most important pixel: index {np.argmax(weights)} (value={pixels[0,np.argmax(weights)]})")
Line by line:

pixels = np.array([[1,0,1,0,1]]) — one sequence of 5 pixel values as a 1×5 matrix. d_model = 5 — dimension equals number of pixels here (each pixel is its own feature). Wq, Wk, Wv = randn(5,5) — each is a 5×5 projection matrix. Maps pixels into query/key/value spaces. Q = pixels @ Wq — project pixels to queries. Shape: (1, 5). K = pixels @ Wk — project pixels to keys. V = pixels @ Wv — project pixels to values. scores = Q @ K.T / sqrt(d_model) — dot product of Q with Kᵀ gives a 1×1 similarity score (since we have 1 sequence). Divided by √5 for scaling. weights = softmax(scores[0]) — convert the single score to attention weights (probability distribution over positions). attn_out = weights @ V — weighted combination of value vectors. loss = np.mean(attn_out**2) — simple L2 regularization-like loss to drive weights toward zero. d_attn = 2*attn_out/attn_out.size — gradient of loss w.r.t. attention output. Wv -= lr * pixels.T @ (weights * d_attn) — update Wv only (simplified). The most important pixel is the one with the highest attention weight.

Q9 — RNN on Pixel Sequences

python
import numpy as np

rows = np.array([[1,1,1],[0,1,0],[1,1,1]], dtype=float)
label = np.array([1.0])
input_size = 3
hidden_size = 4
output_size = 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for lr in [0.01, 0.1]:
    np.random.seed(42)
    Wxh = np.random.randn(input_size, hidden_size) * 0.1
    Whh = np.random.randn(hidden_size, hidden_size) * 0.1
    bh = np.zeros((1, hidden_size))
    Why = np.random.randn(hidden_size, output_size) * 0.1
    by = np.zeros((1, output_size))

    for epoch in range(10):
        h = np.zeros((1, hidden_size))
        for t in range(len(rows)):
            h = np.tanh(rows[t].reshape(1,-1) @ Wxh + h @ Whh + bh)
        y_pred = sigmoid(h @ Why + by)
        loss = np.mean((label - y_pred)**2)
        dy = -(label - y_pred) * y_pred * (1 - y_pred)
        Why -= lr * h.T @ dy
        by -= lr * dy

    print(f"LR={lr} | Loss: {loss:.4f} | Prediction: {y_pred[0][0]:.4f}")
Line by line:

rows = [[1,1,1],[0,1,0],[1,1,1]] — 3 pixel rows of an image, treated as a time sequence (row1 → row2 → row3). label = [1.0] — class label (this pattern is class 1, like a "H" shape). Wxh = randn(3,4) — input-to-hidden weights. Maps 3 pixel values to 4 hidden units. Whh = randn(4,4) — hidden-to-hidden weights. Carries memory from previous time step. This is what makes it an RNN. bh = zeros(1,4) — hidden layer bias. Why = randn(4,1) — hidden-to-output weights. h = zeros(1,4) — initialize hidden state to zeros before processing the sequence. for t in range(len(rows)): — iterate over each pixel row one at a time (sequential processing). h = np.tanh(rows[t] @ Wxh + h @ Whh + bh) — RNN core equation: new hidden state = tanh(current input contribution + previous hidden state contribution + bias). h @ Whh is what gives the RNN its memory. y_pred = sigmoid(h @ Why + by) — after processing all 3 rows, use the final hidden state to make a prediction. dy = -(label - y_pred) * y_pred * (1-y_pred) — gradient of MSE + sigmoid at output. Why -= lr * h.T @ dy — update only Why and by (simplified BPTT). Full backpropagation through time would also update Wxh and Whh.

Q10 — CNN for Cross Pattern Recognition

python
import numpy as np

pattern = np.array([0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0], dtype=float).reshape(1,1,4,4)
label = np.array([1.0])
b_conv = np.zeros(1)
b_fc = np.zeros(1)

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))

def conv2d(x, k):
    h_out = x.shape[2] - k.shape[2] + 1
    w_out = x.shape[3] - k.shape[3] + 1
    out = np.zeros((x.shape[0], k.shape[0], h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            out[0,0,i,j] = np.sum(x[0,0,i:i+2,j:j+2] * k[0,0])
    return out

def maxpool(x):
    h_out = x.shape[2] // 2
    w_out = x.shape[3] // 2
    out = np.zeros((x.shape[0], x.shape[1], h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            out[0,0,i,j] = np.max(x[0,0,i*2:(i+1)*2,j*2:(j+1)*2])
    return out

for lr in [0.01, 0.1]:
    np.random.seed(42)
    kernel = np.random.randn(1,1,2,2) * 0.1
    W_fc = np.random.randn(9, 1) * 0.1

    for epoch in range(10):
        conv_out = relu(conv2d(pattern, kernel) + b_conv)
        pool_out = maxpool(conv_out)
        flat = pool_out.flatten()
        y_pred = sigmoid(flat @ W_fc + b_fc)
        loss = np.mean((label - y_pred)**2)
        d_pred = -(label - y_pred) * y_pred * (1-y_pred)
        W_fc -= lr * flat.reshape(-1,1) * d_pred

    print(f"LR={lr} | Loss: {loss:.4f} | Prediction: {y_pred[0]:.4f}")
    print(f"Activation map:\n{conv_out[0,0].round(3)}")
Line by line:

pattern.reshape(1,1,4,4) — shape is (batch=1, channels=1, height=4, width=4). CNN expects 4D input. kernel = randn(1,1,2,2) — shape (out_channels=1, in_channels=1, kH=2, kW=2). A 2×2 learnable filter. h_out = x.shape[2] - k.shape[2] + 1 — output size = input size - kernel size + 1. For 4×4 input and 2×2 kernel: 4-2+1=3, so output is 3×3. out[0,0,i,j] = np.sum(x[0,0,i:i+2,j:j+2] * k[0,0]) — slide the 2×2 kernel over the input; at each position multiply element-wise and sum. This is the convolution operation. relu(conv2d(...)) — apply ReLU after convolution: negative activations become 0. Helps detect only positive pattern matches. h_out = x.shape[2] // 2 — max pool halves each spatial dimension: 3×3 → 1×1 (integer division). out[0,0,i,j] = np.max(x[0,0,i*2:(i+1)*2,j*2:(j+1)*2]) — take maximum value in each 2×2 region. Keeps the strongest activation, discards weak ones. flat = pool_out.flatten() — convert the 2D feature map to a 1D vector for the fully connected layer. W_fc = randn(9,1) — after conv (3×3=9) → pool (1×1 but shape depends) we have 9 features going to 1 output. y_pred = sigmoid(flat @ W_fc + b_fc) — fully connected layer with sigmoid for binary classification. d_pred = -(label-y_pred)*y_pred*(1-y_pred) — gradient at output. W_fc -= lr * flat.reshape(-1,1) * d_pred — update the fully connected weights.

Q11 & Q12 — Autoencoder for Noise Removal

python
import numpy as np

clean = np.array([[1,1,1,1,0,1,1,1,1]], dtype=float)
noisy = np.array([[1,0,1,1,1,1,1,0,1]], dtype=float)
input_dim = 9
hidden_dim = 4

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

for lr in [0.001, 0.01]:
    np.random.seed(42)
    W_enc = np.random.randn(input_dim, hidden_dim) * 0.1
    b_enc = np.zeros(hidden_dim)
    W_dec = np.random.randn(hidden_dim, input_dim) * 0.1
    b_dec = np.zeros(input_dim)
    losses = []

    for epoch in range(10):
        h = sigmoid(noisy @ W_enc + b_enc)
        output = sigmoid(h @ W_dec + b_dec)
        loss = np.mean((clean - output)**2)
        losses.append(loss)
        d_out = -(clean - output) * sigmoid_deriv(output)
        dW_dec = h.T @ d_out
        db_dec = d_out.flatten()
        d_h = d_out @ W_dec.T * sigmoid_deriv(h)
        dW_enc = noisy.T @ d_h
        db_enc = d_h.flatten()
        W_dec -= lr * dW_dec
        b_dec -= lr * db_dec
        W_enc -= lr * dW_enc
        b_enc -= lr * db_enc

    print(f"LR={lr} | Loss Start: {losses[0]:.4f} | Loss End: {losses[-1]:.4f}")
    print(f"Reconstructed: {output.round(3)}")
Line by line:

clean = [[1,1,1,1,0,1,1,1,1]] — original 3×3 image flattened (ring pattern with center hole). noisy = [[1,0,1,1,1,1,1,0,1]] — same image with 2 pixels flipped (corrupted version). W_enc = randn(x9, 4) — encoder compresses 9 input pixels → 4 hidden units (bottleneck). Forces learning essential features. W_dec = randn(4, 9) — decoder expands 4 hidden units → 9 output pixels. h = sigmoid(noisy @ W_enc + b_enc) — encode the noisy image into a 4-dim latent representation. output = sigmoid(h @ W_dec + b_dec) — decode back to 9-dim reconstructed image. Network learns to ignore noise. loss = mean((clean - output)**2) — compare reconstruction to CLEAN image. This is the key: input is noisy, target is clean, so the network learns to denoise. d_out = -(clean - output) * sigmoid_deriv(output) — gradient at decoder output. dW_dec = h.T @ d_out — gradient for decoder weights. d_h = d_out @ W_dec.T * sigmoid_deriv(h) — backpropagate through decoder to hidden layer. dW_enc = noisy.T @ d_h — gradient for encoder weights. Loss start vs end shows how much the autoencoder improved at denoising. With LR=0.01 it improves faster than 0.001.

Q13 — MLP for Pixel-Based Image Classification

python
import numpy as np

X = np.array([[255,255,255,255],[10,10,10,10]], dtype=float) / 255.0
Y = np.array([[1],[0]], dtype=float)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)

for lr in [0.001, 0.01]:
    np.random.seed(42)
    W1 = np.random.randn(4, 4) * 0.1
    b1 = np.zeros(4)
    W2 = np.random.randn(4, 1) * 0.1
    b2 = np.zeros(1)

    for epoch in range(10):
        a1 = sigmoid(X @ W1 + b1)
        a2 = sigmoid(a1 @ W2 + b2)
        loss = np.mean((Y - a2)**2)
        d2 = -(Y - a2) * sigmoid_deriv(a2)
        dW2 = a1.T @ d2
        d1 = (d2 @ W2.T) * sigmoid_deriv(a1)
        dW1 = X.T @ d1
        W2 -= lr * dW2
        W1 -= lr * dW1

    preds = (a2 > 0.5).astype(int)
    print(f"LR={lr} | Loss: {loss:.4f} | Preds: {preds.flatten()}")

test = np.array([[128,128,128,128]]) / 255.0
a1t = sigmoid(test @ W1 + b1)
a2t = sigmoid(a1t @ W2 + b2)
print(f"[128,128,128,128]: {'Bright' if a2t[0][0]>0.5 else 'Dark'}")
Line by line:

X = [[255,255,255,255],[10,10,10,10]] / 255.0 — two 2×2 grayscale images flattened. Dividing by 255 normalizes from [0,255] to [0,1]. Bright image becomes [1,1,1,1], dark becomes [~0.04, 0.04, 0.04, 0.04]. Y = [[1],[0]] — Bright=1, Dark=0. W1 = randn(4,4) — 4 input features (pixels) → 4 hidden neurons. W2 = randn(4,1) — 4 hidden → 1 output. a1 = sigmoid(X @ W1 + b1) — hidden layer: detects patterns in pixel intensities. a2 = sigmoid(a1 @ W2 + b2) — output: single probability (Bright or Dark). d2 = -(Y-a2)*sigmoid_deriv(a2) — output gradient. d1 = (d2 @ W2.T)*sigmoid_deriv(a1) — propagate to hidden layer. test = [[128,128,128,128]]/255.0 — mid-grey image = [0.502, ...]. Network predicts whether it's closer to bright or dark based on the learned boundary. Decision boundary: roughly at pixel value ~132 (midpoint between 255 and 10).

Q14 — Edge Detection Using CNN

python
import numpy as np

image = np.array([10,10,10,0,0,0,10,10,10], dtype=float).reshape(3,3)
sobel_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
conv_result = np.sum(image * sobel_kernel)
print(f"Manual Sobel result: {conv_result}")

def conv2d_full(img, k):
    h, w = img.shape
    kh, kw = k.shape
    out_h = h - kh + 1
    w_out = w - kw + 1
    out = np.zeros((out_h, w_out))
    for i in range(out_h):
        for j in range(w_out):
            out[i,j] = np.sum(img[i:i+kh, j:j+kw] * k)
    return out

target_output = np.array([[0.0]])

for lr in [0.01, 0.5]:
    np.random.seed(42)
    kernel_learnable = np.random.randn(3,3) * 0.1
    for epoch in range(10):
        feature_map = conv2d_full(image/10.0, kernel_learnable)
        loss = np.mean((feature_map - target_output)**2)
        d_kernel = np.zeros_like(kernel_learnable)
        for i in range(feature_map.shape[0]):
            for j in range(feature_map.shape[1]):
                d_kernel += 2*(feature_map[i,j]-target_output[0,0])*(image[i:i+3,j:j+3]/10.0)
        kernel_learnable -= lr * d_kernel

    print(f"LR={lr} | Loss: {loss:.4f}")
    print(f"Feature map:\n{feature_map.round(3)}")
Line by line:

image.reshape(3,3) — horizontal stripe: top row bright (10), middle row dark (0), bottom row bright (10). Classic horizontal edge. sobel_kernel = [[-1,0,1],[-2,0,2],[-1,0,1]] — Sobel filter detects vertical edges. Left column negative, right column positive: fires when there's a brightness change left→right. conv_result = np.sum(image * sobel_kernel) — manual convolution (single position, 3×3 * 3×3). Shows how much vertical edge response there is. out_h = h - kh + 1 — for a 3×3 image and 3×3 kernel: output is 1×1 (kernel covers the whole image in one step). feature_map = conv2d_full(image/10.0, kernel_learnable) — normalized image (÷10) run through learnable kernel. target_output = [[0.0]] — we want the feature map to be 0 (no response). The kernel learns to produce no activation on this image. d_kernel += 2*(feature_map[i,j]-target)*image_patch — gradient: how much does each kernel weight contribute to the error. The image patch is the "input" that generated this output position. kernel_learnable -= lr * d_kernel — update kernel toward one that gives zero response. With LR=0.5 the kernel changes very aggressively (may overshoot). With LR=0.01 it converges slowly but stably.

Q15 & Q17 — CNN on Digit Patterns (0 and 1)

python
import numpy as np

digit_0 = np.array([1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1],
                   dtype=float).reshape(1,1,5,5)
digit_1 = np.array([0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,1,1,1,1],
                   dtype=float).reshape(1,1,5,5)
X = np.concatenate([digit_0, digit_1], axis=0)
Y = np.array([0, 1], dtype=float)

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))

def conv2d(img, kernel):
    kh, kw = kernel.shape[2], kernel.shape[3]
    h_out = img.shape[2] - kh + 1
    w_out = img.shape[3] - kw + 1
    out = np.zeros((img.shape[0], kernel.shape[0], h_out, w_out))
    for n in range(img.shape[0]):
        for k in range(kernel.shape[0]):
            for i in range(h_out):
                for j in range(w_out):
                    out[n,k,i,j] = np.sum(img[n,0,i:i+kh,j:j+kw]*kernel[k,0])
    return out

def maxpool2d(x, size=2):
    h_out = x.shape[2] // size
    w_out = x.shape[3] // size
    out = np.zeros((x.shape[0], x.shape[1], h_out, w_out))
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for i in range(h_out):
                for j in range(w_out):
                    out[n,c,i,j] = np.max(x[n,c,i*size:(i+1)*size,j*size:(j+1)*size])
    return out

np.random.seed(42)
kernel = np.random.randn(1,1,3,3) * 0.1
conv_out = relu(conv2d(X, kernel))
pool_out = maxpool2d(conv_out)
flat_size = pool_out.reshape(2,-1).shape[1]

for lr in [0.01, 0.1]:
    np.random.seed(42)
    kernel = np.random.randn(1,1,3,3) * 0.1
    W_fc = np.random.randn(flat_size, 1) * 0.1
    b_fc = np.zeros(1)
    for epoch in range(10):
        conv_out = relu(conv2d(X, kernel))
        pool_out = maxpool2d(conv_out)
        flat = pool_out.reshape(2,-1)
        y_pred = sigmoid(flat @ W_fc + b_fc).flatten()
        loss = np.mean((Y - y_pred)**2)
        d_out = -(Y - y_pred) * y_pred * (1-y_pred)
        W_fc -= lr * flat.T @ d_out.reshape(-1,1)
    print(f"LR={lr} | Loss: {loss:.4f} | Pred: {y_pred.round(3)}")

new_pattern = digit_0[0:1]
conv_out = relu(conv2d(new_pattern, kernel))
pool_out = maxpool2d(conv_out)
pred = sigmoid(pool_out.reshape(1,-1) @ W_fc + b_fc)
print(f"New pattern: {'Digit 1' if pred[0][0]>0.5 else 'Digit 0'}")
Line by line:

digit_0.reshape(1,1,5,5) — batch=1, channels=1, 5×5 image. Digit 0 looks like a hollow rectangle (border=1, inside=0). digit_1.reshape(1,1,5,5) — digit 1 looks like a vertical line with base. X = np.concatenate([digit_0, digit_1], axis=0) — stack both into batch of 2. Shape: (2,1,5,5). Y = [0, 1] — labels: digit_0 → class 0, digit_1 → class 1. kernel = randn(1,1,3,3) — one 3×3 filter. Learns to respond to patterns that distinguish 0 from 1. for n in range(img.shape[0]): — process each image in the batch separately. for i,j: — slide kernel across every position of the image. out[n,k,i,j] = sum(img_patch * kernel) — element-wise multiply patch with kernel and sum. h_out = 5-3+1 = 3 — 5×5 input, 3×3 kernel → 3×3 output feature map. maxpool size=2 — 3×3 feature map divided by 2 gives 1×1 (with floor division). So after pool each image is a 1×1 feature map. flat_size = pool_out.reshape(2,-1).shape[1] — compute flattened size dynamically (depends on kernel size). flat = pool_out.reshape(2,-1) — flatten each sample: shape (2, flat_size). y_pred = sigmoid(flat @ W_fc + b_fc).flatten() — predictions for both samples simultaneously. d_out = -(Y-y_pred)*y_pred*(1-y_pred) — gradient at output (shape: 2). W_fc -= lr * flat.T @ d_out.reshape(-1,1) — update fully connected weights. Feature extraction: the kernel learns to detect edges/corners specific to each digit's shape.

Q16 — Variational Autoencoder (VAE)

python
import numpy as np

patterns = np.array([
    [1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1],
    [0,1,0,0,1,1,0,0,0,1,0,0,1,1,1,0],
    [1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0],
    [1,1,1,1,0,0,1,0,0,1,0,0,1,1,1,1]
], dtype=float)

input_dim = 16
latent_dim = 4

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
def sigmoid_deriv(x): return x * (1 - x)

for lr in [0.001, 0.01]:
    np.random.seed(42)
    W_enc  = np.random.randn(input_dim, 8) * 0.1
    W_mu   = np.random.randn(8, latent_dim) * 0.1
    W_logv = np.random.randn(8, latent_dim) * 0.1
    W_dec1 = np.random.randn(latent_dim, 8) * 0.1
    W_dec2 = np.random.randn(8, input_dim) * 0.1

    for epoch in range(10):
        h_enc  = sigmoid(patterns @ W_enc)
        mu     = h_enc @ W_mu
        log_var = h_enc @ W_logv
        eps = np.random.randn(*mu.shape)
        z   = mu + eps * np.exp(0.5 * log_var)
        h_dec = sigmoid(z @ W_dec1)
        recon = sigmoid(h_dec @ W_dec2)
        recon_loss = np.mean((patterns - recon) ** 2)
        kl_loss = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))
        total_loss = recon_loss + 0.001 * kl_loss
        d_recon  = -(patterns - recon) * sigmoid_deriv(recon)
        dW_dec2  = h_dec.T @ d_recon
        W_dec2  -= lr * dW_dec2
        d_h_dec  = d_recon @ W_dec2.T * sigmoid_deriv(h_dec)
        dW_dec1  = z.T @ d_h_dec
        W_dec1  -= lr * dW_dec1
        d_z      = d_h_dec @ W_dec1.T
        d_mu     = d_z + 0.001 * mu
        d_logv   = d_z * eps * 0.5 * np.exp(0.5*log_var) + 0.001*0.5*(np.exp(log_var)-1)
        W_mu    -= lr * h_enc.T @ d_mu
        W_logv  -= lr * h_enc.T @ d_logv

        if (epoch+1) % 5 == 0:
            print(f"LR={lr} | Epoch {epoch+1} | Recon: {recon_loss:.4f} | KL: {kl_loss:.4f}")

    z_sample  = np.random.randn(2, latent_dim)
    h_gen     = sigmoid(z_sample @ W_dec1)
    generated = sigmoid(h_gen @ W_dec2)
    print(f"Generated (LR={lr}):\n{generated.round(3)}")
Line by line:

patterns — 4 digit-like 4×4 binary images flattened to 16-d vectors. W_enc = randn(16,8) — encoder first layer: compresses 16 pixels → 8 hidden units. W_mu = randn(8,4) — maps hidden → mean vector (μ) of latent distribution. W_logv = randn(8,4) — maps hidden → log variance (log σ²) of latent distribution. Using log variance instead of variance keeps values unconstrained and numerically stable. h_enc = sigmoid(patterns @ W_enc) — shared encoder hidden layer for both mu and log_var heads. mu = h_enc @ W_mu — predicted mean of latent Gaussian for each sample. log_var = h_enc @ W_logv — predicted log variance of latent Gaussian. eps = np.random.randn(*mu.shape) — sample random noise ε ~ N(0,1). Shape matches mu. z = mu + eps * exp(0.5 * log_var) — Reparameterization trick: z = μ + ε·σ. This makes sampling differentiable. Note: exp(0.5·log_var) = exp(log σ) = σ. h_dec = sigmoid(z @ W_dec1) — decoder first layer. recon = sigmoid(h_dec @ W_dec2) — reconstructed pattern (16-dim), values in (0,1). recon_loss = mean((patterns - recon)²) — measures how well the decoder reconstructed the input. kl_loss = -0.5 * mean(1 + log_var - mu² - exp(log_var)) — KL Divergence: measures how much the learned latent distribution diverges from N(0,1). Formula: KL(N(μ,σ²) || N(0,1)) = -0.5·Σ(1 + log σ² - μ² - σ²). Forces latent space to be compact and continuous. total_loss = recon_loss + 0.001 * kl_loss — 0.001 is the KL weight (β). Small value so reconstruction dominates early training. d_recon = -(patterns-recon)*sigmoid_deriv(recon) — gradient at decoder output. dW_dec2 = h_dec.T @ d_recon — gradient for decoder output weights. d_h_dec = d_recon @ W_dec2.T * sigmoid_deriv(h_dec) — backprop to decoder hidden. d_z = d_h_dec @ W_dec1.T — gradient reaches the latent z. d_mu = d_z + 0.001*mu — gradient for mu: reconstruction gradient + KL gradient (d(KL)/dmu = mu). d_logv = d_z*eps*0.5*exp(0.5*log_var) + 0.001*0.5*(exp(log_var)-1) — gradient for log_var: reparameterization gradient + KL gradient (d(KL)/d(log_var) = 0.5·(σ²-1)). z_sample = randn(2, latent_dim) — sample 2 new points from N(0,1) in latent space. generated = sigmoid(sigmoid(z_sample @ W_dec1) @ W_dec2) — decode them to get brand-new generated patterns that were never in the training set.

Q18 — Transformer Encoder for Review Classification

python
import numpy as np

X = np.array([
    [[1,0,1,0],[1,1,1,0],[1,0,1,1]],
    [[0,1,1,1],[1,1,0,1],[1,1,1,1]],
    [[0,0,0,1],[0,1,0,0],[0,0,1,0]],
    [[1,0,0,0],[0,0,1,0],[0,1,0,0]]
], dtype=float)
Y = np.array([1,1,0,0], dtype=float)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000**(i/d_model)))
            if i+1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000**(i/d_model)))
    return PE

def layer_norm(x, eps=1e-6):
    return (x - x.mean(axis=-1,keepdims=True)) / (x.std(axis=-1,keepdims=True)+eps)

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0,2,1) / np.sqrt(d_k)
    return softmax(scores) @ V

n_samples, seq_len, d_model = X.shape
PE = positional_encoding(seq_len, d_model)
X_pe = X + PE

for lr in [0.001, 0.01]:
    np.random.seed(42)
    Wq = np.random.randn(d_model, d_model) * 0.1
    Wk = np.random.randn(d_model, d_model) * 0.1
    Wv = np.random.randn(d_model, d_model) * 0.1
    W_ff1 = np.random.randn(d_model, 8) * 0.1
    W_ff2 = np.random.randn(8, d_model) * 0.1
    W_cls = np.random.randn(d_model, 1) * 0.1
    b_cls = np.zeros(1)

    for epoch in range(10):
        Q = X_pe @ Wq
        K = X_pe @ Wk
        V = X_pe @ Wv
        attn_out = attention(Q, K, V)
        X_attn = layer_norm(X_pe + attn_out)
        ff_out = np.maximum(0, X_attn @ W_ff1) @ W_ff2
        X_enc = layer_norm(X_attn + ff_out)
        pooled = X_enc.mean(axis=1)
        logits = (1/(1+np.exp(-(pooled @ W_cls + b_cls)))).flatten()
        loss = -np.mean(Y*np.log(logits+1e-8)+(1-Y)*np.log(1-logits+1e-8))
        d_logits = (logits - Y) / n_samples
        W_cls -= lr * pooled.T @ d_logits.reshape(-1,1)

    preds = (logits > 0.5).astype(int)
    print(f"LR={lr} | Loss: {loss:.4f} | Accuracy: {np.mean(preds==Y)*100:.0f}%")

new_review = np.array([[[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5]]])
new_pe = new_review + PE
Q = new_pe @ Wq; K = new_pe @ Wk; V = new_pe @ Wv
enc = layer_norm(new_pe + attention(Q,K,V))
pred = 1/(1+np.exp(-(enc.mean(axis=1) @ W_cls + b_cls)))
print(f"New review: {'Positive' if pred[0][0]>0.5 else 'Negative'}")
Line by line:

X.shape = (4, 3, 4) — 4 reviews, each with 3 word vectors, each word is 4-dimensional. Y = [1,1,0,0] — first 2 reviews are Positive, last 2 are Negative. K.transpose(0,2,1) — for batched attention (4 samples at once): swap last 2 dims. Shape (4,3,4) → (4,4,3) so Q @ Kᵀ gives (4,3,3) — each sample's attention matrix. scores = Q @ K.T / sqrt(d_k) — for each of 4 reviews: 3×3 matrix of word-to-word attention scores. softmax(scores) @ V — for each review: attention-weighted blend of word vectors. X_attn = layer_norm(X_pe + attn_out) — residual connection: add original input to attention output, then normalize. Prevents information loss. ff_out = max(0, X_attn @ W_ff1) @ W_ff2 — position-wise feed-forward: each word vector goes through the same 2-layer MLP independently. ReLU adds non-linearity. X_enc = layer_norm(X_attn + ff_out) — second residual + norm after feed-forward. pooled = X_enc.mean(axis=1) — Global Average Pooling: average the 3 word vectors into one sentence vector (shape: 4×4). This collapses the sequence dimension. logits = sigmoid(pooled @ W_cls + b_cls) — linear classifier on top: maps 4-dim sentence vector → single probability. loss = -mean(Y·log(p) + (1-Y)·log(1-p)) — Binary Cross-Entropy loss: better than MSE for classification tasks. Strongly penalizes confident wrong predictions. d_logits = (logits - Y) / n_samples — BCE gradient: simply (predicted - actual) / batch_size. W_cls -= lr * pooled.T @ d_logits — update only the classifier head. new_review = [[[0.5,0.5,0.5,0.5],…]] — neutral review (all 0.5) to test generalization. Model predicts based on the learned pattern between positive and negative vector styles.
