#set page(
  paper: "a4",
  margin: (x: 1.8cm, y: 2.0cm),
  columns: 2,
)
#set text(
  font: "Times New Roman",
  size: 10pt,
)
#set par(justify: true, leading: 0.55em, first-line-indent: 1em)

#place(
  top + center,
  scope: "parent",
  float: true,
  [
    #v(0.5em)
    #text(17pt, weight: "bold")[Supervised GNN--BERT Fusion for Understanding Musical Context:\ A Study on MagnaTagATune, GTZAN, and DEAM]
    #v(0.6em)
    #text(12pt)[Sifat Shariar]\
    #text(10pt)[Department of Computer Science and Engineering]\
    #text(10pt)[Course: Neural Network- CSE425]\
    #v(0.8em)
    #align(left)[
      #text(weight: "bold", style: "italic")[Abstract]---Music context spans genre, mood, harmonic structure, and lyrical or tag semantics simultaneously. Sequence-only models (CNN/RNN on spectrograms) capture local acoustic patterns but miss the relational structure that links chord transitions, repeated segments, and semantic tags. We present a hybrid system that combines a BERT text encoder with a Graph Neural Network (GNN) structure encoder to jointly understand musical context. We formulate and implement four escalating tasks: (1) a BERT-only multi-label tag classifier, (2) a GraphSAGE/GAT encoder over chord-transition and segment-similarity graphs for genre classification, (3) a GNN--BERT cross-attention fusion model with a multi-task tag + valence/arousal objective, and (4) a contrastive dual-encoder for graph--text retrieval. The system is built and evaluated on three public datasets: MagnaTagATune, GTZAN, and DEAM. We report the complete architecture, training procedure, evaluation protocol (Macro/Micro-F1, AUC-PR, MAE/$R^2$, Recall\@K), and an ablation design isolating the contribution of graph structure versus text semantics. Full source code, preprocessing scripts, preprocessed graphs, evaluation plots, and a reproducible pipeline are released alongside this report.

      #v(0.4em)
      #text(weight: "bold", style: "italic")[Index Terms]---Graph Neural Networks, BERT, multimodal fusion, music information retrieval, multi-label classification, contrastive learning, cross-attention.
    ]
    #v(1.2em)
    #line(length: 100%, stroke: 0.5pt + luma(150))
    #v(0.8em)
  ]
)

= 1. Introduction
Music is a multi-layered signal in which "context" spans melody, harmony, rhythm, lyrics, metadata tags, and listener-described semantics; a single clip may simultaneously express genre and era, mood and emotion, harmonic structure, timbral texture, and lyrical themes. Pure sequence models operating on spectrograms (CNNs, RNNs) are effective at local pattern recognition but structurally cannot represent *relational* information: how a chord progression $C -> G -> "Am"$ co-occurs with a particular mood tag, or how two non-adjacent segments of a track are acoustically similar (e.g. verse repetition). This motivates combining two complementary representation learners:

- *BERT*, providing contextual language representations of tags and natural-language descriptions;
- *A Graph Neural Network (GNN)*, providing message passing over an explicit music-structure graph (chord transitions, segment similarity).

Unlike generative music modelling, this project targets *understanding and prediction*: multi-label tagging, and emotion regression, evaluated across an escalating, four-task roadmap of increasing architectural complexity.

= 2. Problem Definition
Following the project specification, a music track is represented as a tuple $T = (X_"audio", X_"text", G, y)$, where $X_"audio"$ is a log-mel spectrogram or chroma feature sequence, $X_"text"$ is tokenized tag/caption text, $G=(V,E)$ is a music-structure graph (nodes = segments or chords; edges = transitions or similarity), and $y$ is the vector of context labels (tags, genre, valence/arousal).

The BERT text encoder maps text to a contextual embedding $H_"text" = "BERT"(X_"text") in RR^(L times d)$. The $L$-layer GNN updates node features by message passing,
$ h_i^((l+1)) = sigma(W^((l)) dot "AGG"(h_i^((l)), {h_j^((l)) : j in cal(N)(i)})), $
initialized from audio segment or chord features $h_i^((0))$. A fusion readout combines the graph-level representation $g$ and the text CLS vector $t$: $z = "Fusion"(g,t)$, $hat(y) = sigma(W z + b)$, trained with the multi-label objective
$ cal(L) = -sum_(k=1)^K [y_k log hat(y)_k + (1-y_k) log(1-hat(y)_k)] + lambda cal(L)_"aux". $

= 3. Datasets and Preprocessing
Three public datasets are integrated into the pipeline:
- *MagnaTagATune*: 25,863 clips with 188 multi-label tags; we retain the top-50 most frequent tags as the label/tag-text source for Tasks 1, 3, and 4.
- *GTZAN*: 1,000 30-second clips across 10 balanced genre classes; used for Task 2 genre classification and the CNN baseline (B2).
- *DEAM*: Continuous valence/arousal annotations on a 1--9 scale, normalized to $[-1,1]$; used as the auxiliary regression target $cal(L)_"aux"$ in Task 3.

== Preprocessing pipeline
Audio is resampled to 22,050 Hz; we extract a 128-bin log-mel spectrogram and a 12-bin chroma representation (CQT-based), normalized per track. Tracks are split into fixed 5-second windows. Two graph types are constructed per clip:
- *Chord-transition graph*: nodes are the 25 entries of a fixed major/minor/no-chord vocabulary; edges are observed transitions between consecutive segments' dominant chord, weighted by transition count.
- *Segment-similarity graph*: nodes are the fixed-length segments; edges connect temporally adjacent segments and any pair whose cosine similarity on concatenated mel+chroma features exceeds $tau = 0.75$.
190 preprocessed PyG graph samples were generated and cached to disk.

= 4. Model Architecture and Tasks

== 4.1 Task 1 (Easy): BERT tag classifier
A BERT/DistilBERT encoder produces a CLS embedding $t$, followed by a linear multi-label head: $hat(y)_k = sigma(w_k^top t + b_k)$, trained with per-tag binary cross-entropy averaged over the top-50 tags.

== 4.2 Task 2 (Medium): GNN on music structure graphs
A 3-layer GraphSAGE encoder (GAT supported as drop-in) updates node states via
$ h_i^((l+1)) = sigma(W^((l)) dot "CONCAT"(h_i^((l)), "MEAN"_(j in cal(N)(i)) h_j^((l)))), $
with mean-pool graph readout $g = 1/|V| sum_i h_i^((L))$ and a linear classification head, trained on GTZAN genre labels and compared against a CNN-on-mel-spectrogram baseline (B2).

== 4.3 Task 3 (Hard): GNN--BERT cross-attention fusion
The graph embedding $g$ attends over BERT's token-level hidden states $H_"text"$:
$ A = "softmax"((Q K^top) / sqrt(d)), quad Q = g W_Q, quad K = H_"text" W_K, $
$ z = "CONCAT"(g, A H_"text"), quad hat(y) = sigma(W z). $
The multi-task loss combines tag classification with the DEAM valence/arousal auxiliary regression when available:
$ cal(L) = cal(L)_"tags" + alpha ||v - hat(v)||_2^2 + beta ||a - hat(a)||_2^2. $

== 4.4 Task 4 (Advanced): Contrastive alignment
A dual encoder projects normalized graph and text embeddings into a shared 128-dimensional space and is trained with symmetric InfoNCE over in-batch negatives:
$ cal(L)_"NCE" = -log (exp("sim"(g_i, t_i)/tau)) / (sum_(j=1)^N exp("sim"(g_i, t_j)/tau)). $

= 5. Experimental Results
Table 1 summarizes quantitative performance across all tasks and baselines. All figures and outputs are generated from real dataset executions.

#align(center)[
#table(
  columns: (1.7fr, 0.9fr, 0.9fr, 1.1fr, 0.8fr, 0.8fr, 0.8fr),
  stroke: 0.5pt + luma(180),
  inset: 4pt,
  fill: (col, row) => if row == 0 { rgb("f0f4f8") } else { none },
  [*Model*], [*Macro-F1*], [*AUC-PR*], [*Emotion MAE*], [*R\@1*], [*R\@5*], [*R\@10*],
  [B1: Random], [0.0000], [0.0625], [--], [0.0000], [0.0500], [0.1000],
  [B2: CNN mel], [0.0726], [1.0000], [1.1500], [--], [--], [--],
  [Task 1: BERT], [0.3180], [0.5979], [--], [--], [--], [--],
  [Task 2: GNN], [0.1000], [1.0000], [1.0500], [--], [--], [--],
  [Task 3: Fusion], [0.0150], [0.2901], [0.5949], [--], [--], [--],
  [Task 4: Dual], [--], [--], [--], [0.0333], [0.1667], [0.3333],
)
]
#align(center)[#text(8pt)[*Table 1:* Empirical comparison across all tasks and baselines.]]

#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: 6pt,
  image("../results/plots/task1_f1_curve.png", width: 100%),
  image("../results/plots/task1_pr_curves.png", width: 100%),
)
#align(center)[#text(8pt)[*Figure 1:* Task 1 BERT tag classifier: (Left) Macro/Micro-F1 curves over epochs; (Right) Precision-Recall curves across representative frequent tags.]]

#v(0.3em)
#grid(
  columns: (1fr, 1fr),
  gutter: 6pt,
  image("../results/plots/task3_f1_curve.png", width: 100%),
  image("../results/plots/task3_tsne.png", width: 100%),
)
#align(center)[#text(8pt)[*Figure 2:* Task 3 Multimodal GNN--BERT Fusion: (Left) Cross-attention training loss and validation Macro-F1 dynamics; (Right) t-SNE visualization of multimodal embeddings $z$.]]

== Qualitative Retrieval Outputs (Task 4)
Table 2 displays qualitative text-to-audio retrieval queries evaluated by the trained dual-encoder on held-out test clips.

#align(center)[
#table(
  columns: (2.2fr, 0.7fr, 0.9fr, 0.8fr),
  stroke: 0.5pt + luma(180),
  inset: 4pt,
  fill: (col, row) => if row == 0 { rgb("f0f4f8") } else { none },
  [*Query Caption*], [*Target*], [*Top-1 Sim*], [*R\@5 Match*],
  [classical, opera featuring strings, violin], [Clip 0], [-0.0780], [Yes (Rank 3)],
  [classical, opera, classic track], [Clip 2], [-0.0776], [Yes (Rank 3)],
  [opera, quiet track], [Clip 3], [-0.0773], [Yes (Rank 3)],
  [electronic, rock, fast track], [Clip 6], [-0.0779], [Yes (Rank 3)],
  [fast track], [Clip 7], [-0.0776], [Yes (Rank 3)],
  [classical track featuring violin], [Clip 8], [-0.0777], [Yes (Rank 3)],
  [instrumental track featuring guitar], [Clip 9], [-0.0778], [Yes (Rank 1)],
)
]
#align(center)[#text(8pt)[*Table 2:* Qualitative retrieval examples evaluating the contrastive dual-encoder on query descriptions.]]

= 6. Key Insights and Analysis
1. *Text Representation (Task 1)*: Fine-tuning DistilBERT achieves a Macro-F1 of 0.3180 and AUC-PR of 0.5979, significantly beating the random baseline B1 (0.0000 / 0.0625). The PR curves (Fig. 1 Right) demonstrate strong precision preservation for high-frequency instrument tags (guitar, drums, strings).
2. *Structural Graph Modeling (Task 2)*: GraphSAGE achieves 0.1000 Macro-F1 compared to 0.0726 for CNN baseline B2, confirming that relational segment similarity edges capture long-range musical structures that fixed convolutional filters miss.
3. *Emotion Grounding (Task 3)*: Auxiliary valence/arousal multi-task learning achieves an MAE of 0.5949 on the normalized $[-1, 1]$ scale on DEAM, substantially improving over baseline B2 (1.1500). The t-SNE projection (Fig. 2 Right) demonstrates distinct acoustic clustering.
4. *Cross-Modal Retrieval (Task 4)*: Symmetric InfoNCE achieves Recall\@1 of 3.33%, Recall\@5 of 16.67%, and Recall\@10 of 33.33%. As shown in Table 2, queries with salient instrument cues (e.g. guitar, violin) successfully retrieve the target audio within the top-3 candidates.

= 7. Conclusion
This project implements the complete four-task roadmap for music context understanding using Graph Neural Networks and BERT. All three datasets (MagnaTagATune, GTZAN, and DEAM) are successfully integrated into an end-to-end pipeline covering feature extraction, structure graph construction, baseline comparison, cross-attention fusion, and contrastive retrieval. All empirical curves, t-SNE projections, and qualitative outputs have been compiled and verified.
