# Literature Review — Speech Deepfake Detection

## Papers

### 1. A Survey on Speech Deepfake Detection — https://arxiv.org/pdf/2404.13914 — [Li et al, 2025] 

**I. Intro**  
- systematic review of 200+ papers up to 03/2024
    - covers TTS and VC deepfakes, not "physical" (recording playback)
    - also covers partial deepfakes
- 
**II. Datasets and Evaluation**
- Many datasets contain noise or other artifacts (compression), different languages
- EER standard, AUC not mentioned
- For purpose of this project will ignore Partial Deepfake Sections

**III. Fully Deepfake Detection**
- Covered in here: model architecture, training optimizations, non-ML methods
- most recent models contain front end (feat extraction) and backend (clf)
    - E2E models gaining attn
    
**III.I Feature Eng**
- 3 categories: hand-crafted spectral, DL, "other analysis-oriented" features

**III.I.I Hand-crafted Features**
- Magnitude-based spectral coef
    - just about every transform imaginable (EXCEPT WAVELETS) alongside first and second order derivatives
    - short and long term windows
- Phase ""
    - deepfake often lacks natural phase info
    - whether net positive or performance improved not reported
- Bispectrum
    - Bispectrum: fft(3rd order cumulant), analyzes nonlin interactions in signals
    - First 4 moments of bispectral correlation (?) difference between fake and bonafide (high SNR constraint)
- Spectrogram
    - Mel, CQT, STFT, complex CQT
    - Texture analysis tools: local binary pattern (LBP), modified local ternary pattern (MLTP)

**III.I.II DL Features**
- Filter-learning
    - SOTA/baseline here is prob RawNet2 which uses SincNet, whereby filter f_c's are learned and dims are expanded such that output front-end is time-freq
- Deep embedding derived from hand-crafted feat
    - Some front ends compute fixed features but then encode those via DNN (various architectures eg RNN, ResNet)
    - Most recent example: CQT-spec -> U-Net with Attn & skip connections (unclear whether at bottleneck or during encoding)
- Pretrained embedding: 
    - Self supervised learning (SSL) (all learned representations of raw waveforms, so forget hand-crafted feat) representations outperform in deepfake detection compared to models with fixed/handcrafted front ends
        - e.g. wav2vec 2.0
    - Fine-tuning / Enhancing SSL Feat:
        - learned norm params applied to hidden states of transformer layer in wav2vec2.0
        - adding attention into SSL models
        - pre-trained SSL models trained on many languages (tons of data) help in non-english deepfake detection
    
**III.I.III Other analyis oriented features**
- Prosody and semantic feat:
    - Prosody: voice production features (DL-based)
    - Semantic: emotion recognition (DL-based); work on TTS-generated fake speech as opposed to VC
- Silence:
    - TTS-gen: duration proportion of silence
    - VC-gen: content of silence (fake has discontinuities)
    - Using VAD to remove silence degrades detection performance
    - Research on silence conducted using clean data (high SNR)
- Freq subband feat:
    - Basically only attending to parts of spectrum
    - 0-4 kHz contains voiced spectral differences
    - 4-8 kHz contains unvoiced/silent spectral differences
    - That said, 0-400 Hz can still achieve good performance (focus only on speech f0)
- Varied input length:
    - Zpadding/truncating can lead to info loss/propagation of bad info: pooling layer proposed to precede DL models to handle variable length inputs (how?) which outperforms fixed length inputs w same back end clf (does this only apply to eng features?)
- Other directions (alternative feat):
    - just high freqs
    - energy loss between words
    - using compression metadata - could be similar to one of my ideas - Yadav et al
    - expand mono to stereo then process channels separately? ...in what way(s)

**III.I.IV Performance discussion on feat selection**
- moving away from engineered features toward deep embedding representation (pretrained SSL). Why?
    - learnable features more likely to capture things overlooked by fixed mathematical features
    - engineered features differ per language (learned feat, especially SSL embeddings trained on huge datasets generalize better)
- hand-crafted features cheap compute: suggest using combination of engineered feat and learnable features

**III.II Classifier Architecture**
- ML approaches dominate: common to have multiple architectures in clf

**III.II.I Trad ML Clf**
- support vector machines, gaussian mixture (GMM), and random forest all used in early DFD
- GMM fundamental baseline for ASVspoof

**III.II.II CNN Clff**
- Not much here...
- LCNN (light cnn), use MFM (max feat map) activation instead of ReLU
- Due to freq domain loss, suggested splitting spectrogram into 3 freq bands and running each through a CNN

**III.II.III ResNet**
- Enhancement: squeeze & excitation (SE) blocks w ResNet, forms SE-Net (dynamic channel-wise feat recalibration)
- Enhancement: ResMax - using MFM with ResNet blocks
- Enhancement: DenseNet - introduces skip connections (w ResNet?)
- Res2Net: variant of ResNet where bottleneck is modified; channels segmented and then prev layer features added before conv operation

**III.II.IV Graph Neural Net (GNN)**
- Specifically mentioned: GCN, GAT
- Sounds common to split spectrogram into grid sections, get feat embeddings using CNN, which pass through layers of GCN (graph conv network)
- GIN used at all for global attention?
- If most promising method, how does param count compare to runners up?

**III.II.V Transformers**
- Usually coupled with ResNet or CNN
- Compact Conv Transformer (CCT):
    - spect -> 2 2D conv layers -> transf
    - instead of spect patches -> transf
    - OCT enhances CCT w 1D conv and fewer transf encoders
- Rawformer:
    - SE-Res2Net w positional agg -> transf
    - aim: local and global dependencies
- Trasf typically temporal only but recent work can attend to both time/freq with transf

**III.II.VI Time Delay NN (TDNN)**
- typical in speech recog (SR) 
- performance in DFD seems suboptimal

**III.II.VII Diff Architecture Search (DART)**
- Method for determining best architecture?
- Meta to actual architecture methods themselves?

**III.II.VIII Other Architectures**
- Quantum NN

**III.II.IX Pooling and Attention Mechanism**
- Statistical pooling: 
    - general case uses mean and std of frame level embeddings
    - ASP: introduces attention to stat pooling
- Attention:
    - Attention integrated into clf like LCNN and ResNet
    - Well known mechanisms: CBAM, DANet

**III.II.X Performance Discussion on clf selection**
- Clf needs to extract local and global patterns
    - May require ensembling, e.g. CNN + transf; CNN + GAT
- From table 4 (clf performance comparison)
    - CNN, ResNet, GNN, Transf in some ensemble most likely optimal
    - Must consider complexity, overfitting, global relationships

**III.III E2E Architecture**
- E2E architectures gaining attention
- Feat extraction stage can lose info irrevocably
- Some alleged E2E architectures still use learnable SincNet (BPF fcs)
- Noted work using 1D conv
    - some with single small or larger kernel size
    - eventually some have tried multi conv layers (presumably with diff kernel size)
        - Wu et al [178] - "genuinization trans", similar to autoencoder
    - Integration w Res2Net blocks (additional treatments)

**III.IV Training Optimization Techniques**
- Separated into Data Aug (DA), Loss Func, Activation Func

**III.IV.I DA**
- Masking, Mix-up, Codec Variation, +...

- Masking:
    - SpecAugment: randomly mask blocks of freq bins/time steps/channels on spect or feat maps
        - Improves DFD performance
    - SpecAvg uses avg feat map value rather than zeroing
- Mix-up:
    - SpecMix: mixing spect data from one example with another to create new examples
        - new labels require weighted avging of orig examples
    - Prevents overfitting
    - How much of training dataset to specmix? All or none degrades DFD performance
        - Does [36] mention optimal approach?
- Noise addition:
    - Robustness improved adding noise or reverb (RIRs)
    - RawBoost (improves reliability DFD on ASVspoof2021 LA set)
- Codec aug:
    - Categories: Multimedia or Transmission encoding (all impose info loss)
        - Multimedia looks like just resampling then resampling back to 16k?
        - Transmission: actual VOIP or tele codecs w diff bitrates
        - Packet loss
        - Simple BPF to mimic speech codec
- Other:
    - Speed perturbation (?)
    - Time stretching
    - Pitch shifting
    - Gen of new DF using vocoders

- Discussion:
    - Common to combine DA techniques!
    - Table 5 shows Noise and Codec Aug methods help the most
    - Results show that DA effectiveness can be dataset and feature dependent
        - open area for work: correlating performance to model architecture and DA choices

***III.IV.II Loss Func**
- Most common: Cross Entropy (CE) w Softmax, but others proposed / tailored for specific tasks

- CE w Softmax
    - pdf over 2 classes (real or fake)

- Margin-based CE loss funcs:
    - Additive Margin, Large Margin Cosine, One Class Softmax + mods
    - Generally try to max inter-class var while min intra-class var

- Alternatives:
    - MSE:
        - Margin-based loss sensitive to hyperparam settings
        - MSE is hyperparam free
    - Focal
        - common when data imbalance exist in training data
        - mult factor to standard CE loss
    - Center loss
        - integrated alongside CE to min intra-class var
        - helpful bc CE focuses on max'ing inter-class var

    - Discussion
        - OC-Softmax highly effective
            - prevents overfitting on known attack types while generalizing well to unknown
        - Hybrid loss functions promising

**III.IV.III Activation Function**
- Learnable AFs outperform (e.g. parametric or attention relu)

**III.V Non ML Based Dectection Models**
- Vocal tract model
- Voicefox (STT less accurate on deepfakes)
- Generalizability to range of deepfake modes questionable

**IV Training and Robustness Advancements in Fully Deepfake Detection**
- Improvements have been made in robustness, efficiency, interpretability

**IV.I Training Strategies**
- Improvements aim at model complexity, data scarcity, computational efficiency
- e.g. Siamese network w LCNN, ResNet, SE-Net
    - enhance learned representations without driving up parameter count
    - handles dataset imbalance
- Low Rank Adaption (Lora)
    - transfer learning method where two low rank matrices introduced to existing model
    - prevents forgetting
    - has been used to generate new deep embeddings on multi head component of wav2rec 2.0 transformer
- Knowledge Distillation
    - large models difficult to deploy on edge devices
    - large model is teacher, smaller deployment model is student: try to get student to perform as teacher

**IV.II Interpretability of results**
- Tools for explainable AI in DFD:
    - SHAP scores, Deep Taylor, Layerwise Relevance Program (LRP), Grad-CAM, activation maps

**IV.III Defense to adversarial attacks**
- Adversarial attacks: Fast Gradient Sign Method (FGSM), Projected Grad Descent (PGD)
- In response, adversarial training
- Suggested data augmentation: BPF, Denoising
- Knowledge Distillation also used

**IV.IV Attacker source tracing**
- Neural embeddings (w RNNs) help characterize attacker signatures, vocoders, etc
- New DFD challenges feature deepfake alg detection
    - KNN and center-based similarity max methods used

**IV.V Robustness on cross dataset**
- Cotraining: using different datasets all as training data
- Cross dataset eval: test generalization

- Multi-dataset Cotraining / Continual learning
    - combining data from different domains does not guarantee generalizability
    - RAWM, RWM: help when existing model fine tuned with out of distribution data (OOD)
- Cross-dataset evaluation
    - models trained on ASV2019 dataset failed badly on ITW (in the wild) dataset
        - could be due to channel effect mismatch between datasets - what does this mean?
        - additional components added to address

**V Integration of ASV to Deepfake Countermeasures**
- SASV - spoofing aware speaker verification
- Associated challenges (SASV challenge) use specific metrics
- How is ASV integrated if user's data is required (if ASV outputs similarity score to verify match)
**V.I SOTA for SASV**
- Cascaded systems
    - concat of ASV and CM systems; order seems to matter
    - both binary clf?
- Score Level Fusion
    - scoring subsystems (ASV / CM) and combining scores
- Feat Embedding Level Fusion
    - Either concat of embeddings from ASV and CM subsystems or integrated embedding extraction
- Integrated E2E system
    - research suggests this method (rather than separate subsystems) can make up for lack of Deepfake data or dataset imbalance

**V.II Performance discussion on SASV models**
- Data suggests ensemble methods outperform integrated


### 2. AASIST: AUDIO ANTI-SPOOFING USING INTEGRATED SPECTRO-TEMPORAL GRAPH ATTENTION NETWORKS — https://arxiv.org/pdf/2110.01200 — [Jung et al, 2021] 

**I Introduction**
- Two major scenarios being studied: LA and PA (logical and physical access)
    - This paper focuses on LA
    - LA includes TTS and VC type attacks
- "Discrimitive Info" ie spoofing artifacts lie in spectral and temporal domains
- Prev work by these authors use RawNet2-like encoder and GATs to model heterogeneous (spectral and temporal)
    - E2E systems
    - two parallel graphs w .* on the two graphs
- Extensions proposed:
    - HS-GAL: heterogeneous stacking graph attn layer
        - concurrently model both spectral and temporal graph representations
    - Max Graph Operation (MGO)
        - mimicks max feature map (MFM)
        - two branches, each w 2 HS-GAL -> Graph pooling -> element-wise max
            - idea is diff branches learn diff artifacts
    - Readout scheme using stack node (?)
    - AASIS-L (85k param lightweight version)

**II Preliminaries**
- section describes encoder (RawNet2-like)
- Graph module: attention and pooling
- Only the first few blocks in diagram (up to graph module blocks) in Fig 1

**II.I RawNet2-based Encoder**
- Diff from RawNet2 in that 2D img w 1 ch used (output of sinc conv layer)
- Architecture: 6 resid blocks w pre-activation (norm before activation)
    - each block: BN -> conv2 -> SeLU -> MaxPool

**II.II Graph Module**
- Graph Attn Network
    - GATs being applied to both speaker verification and anti-spoof
    - In this application, graphs are fully connected (edge between each and every node pair)
        - relevance of node pairs learned through attn weights
- Graph Pooling
    - scale down graph, reduce complexity, improve discrimination
    - top k values retained after projection, mult by sigmoid

**III AASIST**
- see fig 1 for architecture

**III.I Graph Combination**
- Two graphs in par: one for spectral one for temporal info
- obtained by taking row/col-wise maxes from F matrix which is encoder output
- Gs and Gt are resulting graphs
- Gst (combined graph) formed by adding edges between every node in Gs and Gt
- Allows deriving attention weights between pairs of heterogen nodes (spectral / temporal)

**III.II HS-GAL**
- Hetero Stacking Graph Attn Layer
- 2 components: hetero attn and stack node
- Hetero Attn:
    - 3 different projection vectors used to compute hetero attn weights
        - Gs to Gs, Gs to Gt and Gt to Gs, and Gt to Gt
- Stack Node
    - role: accumulate hetero info
    - behavior similar to clf tokens except node connections are unidir
    - !!Requires further understanding!!

**III.III MGO and readout**
    - MGO = 4 HSGAL and 4 graph pooling layers total
    - !!Requires further understanding!!

**III.IV Lightweight variant AASIST-L**
    - identical architecture to AASIST, but #params reduced to 85k
    - Population-based training
    - Outperforms all models except AASIST

**IV Experiments and Results**
**IV.I Dataset and metrics**
- Dataset: ASVspoof 2019 LA
    - 3 subsets: train, dev, eval
    - Train and dev sets contain DFs from 6 spoofing algs
    - Eval set contains DFs from 13 spoofing algs 
- Metrics used: t-DCF and EER
    - EER is the DFD-relevant metric
    - random seeds affect performance 

**IV.II Implementation Details**
    - input raw waveforms 64600 samples long
    - Adam optim w LR=1e-4 + cosine annealing LR decay

**IV.III Results**
    - AASIST vs SOTA
        - 5/6 top performers work on raw waveform input
        - top 3 are all GAT
    - AASIST-L
        - w half precision inference (using 16bit float rather than 32) and pruning, could work on embedded systems
    - Ablations
        - heterogenous attn had most performance impact
        - MGO also important


### 3. E2E Anti-Spoofing with RawNet2 - https://arxiv.org/pdf/2011.01108 - Tak et al
#### [Architecture sections only: 2 & 3]

**II Previous Work**
- orig RawNet work for automatic speaker verification (ASV)
- E2E defined as simultaneous and joint optim of every component in the model
- RawNet architecture
    - CNN that outputs speaker embeddings
    - Operates on raw waveform
    - Resid blocks used
    - LSTM or GRUs used
    - Completely unconstrained first layer slows training, hence SincNet learned BPF params
- RawNet2
    - SincNet -> ResBlocks -> GRU
    - New: feature wise map scaling (FMS)
        - attn mechanism whereby res block outputs passed through sigmoid
    - Embedding dim increased from 128 to 1024 in RawNet2
    - Cosine Similarity used for clf as opposed to DNN backend
- Results show E2E architectures can improve ASV performance: can RawNet2 improve DFD?

**III Application to Anti-Spoofing**
- Modifications to RawNet2 for DFD:
    - No layer norm on waveform inputs
    - BPF params not learned..? 
    - Filt length changed from 251 to 129
    - Incr #filts to 512 in 2nd res block
    - GRU (1024 nodes) -> FC -> softmax
- Training:
    - ADAM optim
    - LR = 1e-4
    - 100 epochs
    - 32 mini batch size



---

## Synthesis

### Architectures
### Loss Functions
### Generalization & Dataset Shift
### Evaluation Metrics & Protocols
### Open Questions