# Modular relational graphs, learning digits

An interactive, single-file web demo of how a **relational graph with community structure** becomes the wiring of a multilayer perceptron (MLP), and how that network learns to classify handwritten digits by backpropagation, entirely in the browser.

It is a companion to:

> Yash Arya and Sang Hoon Lee, “Effects of relational graph modularity and depth on the learning performance of neural networks,” *Journal of the Korean Physical Society* (2026).
> DOI: [10.1007/s40042-026-01730-5](https://doi.org/10.1007/s40042-026-01730-5)

The demo was created by Claude Opus 5.5.

## Running it

Open `index.html` in any modern browser. There is nothing to install or build: the page is fully self-contained, including the dataset, and trains the networks in plain JavaScript on your machine.

To host it with GitHub Pages, push `index.html` to the repository and enable Pages in the repository settings; the demo is then served at the repository’s Pages address.

The only external request is for the Schibsted Grotesk font from Google Fonts. Without a network connection the page falls back to the system font and works the same.

## What the demo shows

**Relational graph.** Node *i* stands for neuron *i* in every hidden layer. A link *i–j* means neurons *i* and *j* exchange messages, and every node has a self-loop. Hover a node or a link to see exactly which weights it turns into. “Play message passing” animates the messages sent along each link in every layer.

**Weight matrix.** For the selected layer *r*, cell (*i*, *j*) shows the learned weight *w<sub>ij</sub><sup>(r)</sup>* from neuron *j* to neuron *i*. It is the graph’s weighted adjacency matrix with self-loops on the diagonal. Missing links are fixed at zero, and communities show up as blocks on the diagonal.

**From the relational graph to the whole network.** The full MLP: 64 pixels feed hidden layer 1 densely, 4 layers of message exchange on the graph connect the 5 hidden layers, and hidden layer 5 feeds the 10 digit outputs densely. Hovering anywhere traces a neuron or link through every layer.

**Input and prediction.** Pick a test digit or draw one, and compare the output probabilities of the modular network with those of the fully connected baseline.

**Test error while training** and **Many runs, same graph.** Training starts automatically and repeats for K independent runs (default 50, up to 100). Each run starts from new random weights on the same fixed graph, trains for 30 passes, and pairs the modular network with a fully connected network that shares its starting weights and batches. The runs panel reports mean ± standard deviation, how many runs each network won, and the mean paired difference with its standard error.

**Does mixing matter?** Retrains the current setup at six values of the mixing parameter μ, five graphs each, in the spirit of Fig. 7 of the paper.

## How the translation works

Each layer of message exchange follows Eq. (1) of the paper:

```
x_i^(r+1) = ReLU( Σ_{j ∈ N(i)} w_ij^(r) x_j^(r) )
```

where the neighbourhood N(*i*) includes *i* itself (the self-loop). In terms of the MLP:

- **Node *i*** is neuron *i*, present in every hidden layer. Its self-loop becomes *w<sub>ii</sub>*, the connection from neuron *i* in one layer to neuron *i* in the next.
- **A link *i–j*** becomes two weights per layer, *w<sub>ij</sub>* and *w<sub>ji</sub>*.
- **A missing link** fixes both of those weights at zero in every layer.
- **Depth** is the number of hidden layers. With 5 hidden layers there are 4 graph-shaped weight layers; each has its own weights but the same wiring.
- The input and output connections are dense and not shaped by the graph, as in Fig. 1(a) of the paper.

The fully connected network, the complete graph, is the baseline throughout.

## Graph generation

Graphs follow the paper’s simplified LFR benchmark:

1. The N requested nodes are split into *c* equal communities.
2. Each community is built as an Erdős–Rényi graph with link probability *p*, or as a static scale-free graph (Goh, Kahng and Kim) with degree exponent γ and average degree *m*.
3. Each community is trimmed to its largest connected component (footnote 2 of the paper), so the number of neurons per layer can be smaller than N.
4. A fraction μ of all links is rewired so they join different communities. Rewiring swaps link endpoints, which keeps every node’s degree and the total number of trainable weights fixed.

The page reports the measured Newman–Girvan modularity Q next to the expected value Q ≈ (1 − μ) − 1/*c*.

## Controls

| Control | Range | Meaning |
|---|---|---|
| Graph model | ER or scale-free | Model used inside each community |
| Nodes requested, N | 6–40 | Nodes before trimming to largest components |
| Communities, *c* | 1–4 | Number of equal-sized communities |
| Link probability, *p* | 0.2–1 | ER only |
| Degree exponent, γ | 2–6 | Scale-free only |
| Average degree, *m* | 2–5 | Scale-free only |
| Mixing, μ | 0–1 | Fraction of links rewired between communities |
| Independent runs, K | 5–100 | Number of repeated training runs |
| Learning rate | 0.001–0.03 | Adam step size |

Changing any graph setting draws a new graph and restarts the runs automatically.

## Training details

- **Data:** the UCI Optical Recognition of Handwritten Digits set (not MNIST); see [Training and test data](#training-and-test-data).
- **Network:** 64 inputs, 5 hidden layers of N neurons with ReLU, and 10 softmax outputs.
- **Optimisation:** cross-entropy loss, Adam, mini-batches of 32, 30 passes per run. Weights are initialised with He scaling based on each neuron’s actual number of inputs, and masked weights stay exactly zero.
- **Metric:** top-1 test error on the 300 held-out digits.

## Training and test data

**This demo does not use MNIST or CIFAR-10.** It trains on the *Optical Recognition of Handwritten Digits* dataset from the UCI Machine Learning Repository, in the preprocessed version bundled with scikit-learn as [`sklearn.datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

| | This demo | MNIST, for comparison |
|---|---|---|
| Images | 1,797 | 70,000 |
| Resolution | 8×8 (64 inputs) | 28×28 (784 inputs) |
| Pixel values | 0–16, scaled to 0–1 | 0–255 |
| Writers | 43 | about 500 |

- **Preprocessing (by the dataset authors):** each 32×32 bitmap was divided into non-overlapping 4×4 blocks, and the inked pixels in each block were counted, giving an 8×8 image with values from 0 to 16.
- **Split used here:** the 1,797 images are shuffled once with a fixed seed (NumPy `default_rng(0)`). The first 1,497 are the training set and the last 300 the test set. All reported error rates are top-1 error on those 300 test images.
- **Embedding:** the whole set is stored in the HTML file (about 115 kB, one character per pixel), so the page never downloads data.
- **Drawn digits:** digits drawn on the page are cropped, scaled to 32×32 and reduced to 8×8 in the same block-counting way.
- **Why not MNIST:** a self-contained page cannot download data, and the full MNIST set is too large to embed in a single HTML file.

### Data references

- E. Alpaydin and C. Kaynak, “Optical Recognition of Handwritten Digits,” UCI Machine Learning Repository (1998). DOI: [10.24432/C50P49](https://doi.org/10.24432/C50P49). License: CC BY 4.0.
- F. Pedregosa *et al.*, “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research* **12**, 2825–2830 (2011).

## Disclaimer

This demo is a visual illustration of the method, not a replication of the paper’s results. The paper used CIFAR-10, 128-node graphs and 200 epochs on a GPU; this demo uses the much smaller UCI digits set and a smaller network, so its numbers are noisy and needn’t match the paper’s. In informal runs with the default settings, the modular and fully connected networks were within noise of each other.

CIFAR-10 reference: A. Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” Technical Report, University of Toronto (2009), <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>.

## Related code from the paper

- Simplified LFR benchmark generator: <https://github.com/yasharyaa/Simplified_LFR_Benchmark_Graph>
- Relational graph web visualization: <https://github.com/yasharyaa/relational_graph_web>
- graph2nn (You, Leskovec, He and Xie, 2020): <https://github.com/facebookresearch/graph2nn>
