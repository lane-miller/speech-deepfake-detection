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


### 3. E2E Anti-Spoofing with RawNet2 - https://arxiv.org/pdf/2011.01108 - Tak et al 2021
#### [Architecture sections only: 2 & 3]

**I Introduction**
- Recent ASV Research:
    - Artifacts from VC or TTS exist at subband level (low freqs?)
    - High spectral res front ends help, even if paired with a simple backend clf
    - e.g. A17 attack mode: artifacts captured but not detected if models trained on typical training data
        - if trained on eval data, can detect
- Can unseen attacks be detected without representative training data?
- OC (one class) clf trained only on bona fide data
    - successful broadly but still fail on ASVspoof 2019 data
- Assumption is that non-hand crafted features help

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

### 4. Does Audio Deepfake Detection Generalize? - arXiv:2203.16263 - Muller et al 2026
- Describes implementation of RawNet2 to DFD / anti-spoofing

**Abstract**
- Why are successful models successful?
- Authors do a systematic review of 12 architectures to
    - find what contributes to success
    - report on generalization 
- New celeb deepfake dataset poses significant challenge to SOTA
    - reserach to tied to ASVspoof?

**I Introduction**
- Rehash of abstract

**II Related Work**
**II.I Model Architectures**
- Overview of models used for evaluation 
- LSTM Models
    - 3 LSTM layers -> linear layer -> avg over time dim => embedding
- LCNN
    - Light CNN: use max feat map activations
    - Attn or LSTM also added
- MesoNet
    - From vid deepfakes
    - 4 CNN layers -> BN -> MaxPool -> FC clf
- MesoInception
    - extension of above with inception blocks
- ResNet18
    - implementation of ResNet as expected (CNN + skips)
- Transformer
    - 4 self attention layers, 256 hidden dim + skips
    - positional encodings for time
- CRNNSpoof
    - 1D conv + recurrent layers
    - works directly on waveform, E2E
- RawNet2
    - E2E, uses SincNet
- RawPC
    - Also uses SincNet
    - architecture determined by diff arch search
- RawGAT-ST
    - E2E graph attn network (predecessor to AASIST?)

**III Datasets**
- Training and Eval w ASVSpoof 2019 LA
    - fakes are from 19 TTS algs
- Authors also created dataset of celeb deepfakes + real found footage

**IV Experimental Setup**
**IV.I Training and Evaluation**
**IV.I.I Hyperparams**
- All models trained using
    - CE loss
    - Adam optim
    - LR = 1e-4 w scheduler
    - 100 epochs, early stopping patience = 5
**IV.I.II Train/eval split**
- Use train and dev for training (ASV dataset)
- Eval: eval split from ASV and in the wild ITW (out of domain)

**IV.I.III Evaluation Metrics**
- EER and tDCF but only EER for ITW eval (tDCF req false alarm and miss cost)

**IV.II Feature Extraction**
- for models requiring preproc, spectrograms (513 dim) created and used from:
    - constant Q transform: cqtspec
    - log: logspec
    - mel-scaled: mel-spec
- Otherwise raw audio used as input

**IV.III Audio Input Length**
- Some models don't allow var input length
    - global avg pool layer used to fix this (but goes on to describe fixed 4s inputs??)
- 4 s fixed input length
    - if longer, choose random 4s segment
    - if shorter, repeat
    - for eval, inputs always at least 4s

**V Results**
- Table 1 holds results
- Models not fine tuned
**V.I Fixed vs Var input length**
- Fixed input length severely degrades performance
- Recommended to input full audio examples
**V.II Effects on Feat Extraction Techniques**
- E2E (learn feat) > Hand Feat
- For spec: cqt and log > mel (by a lot)
**V.III Evaluation on ITW data**
- Performance degrades wrt ASVspoof data by 200-1000%
- Best model, RawNet2 retrained including eval data from ASVspoof -> no improvement on ITW

### 5. ASVSpoof 5 ... - https://arxiv.org/pdf/2502.08857 - Wang et al 2025
**Dataset/Partition Sections: 2.1, 2.2, 3, 5**
**II.I Database Generation**
- Earlier ASVspoof used VCTK (clean high quality conditions)
- ASVspoof 5 uses MLS English dataset: ~5000 speakers in variety of acoustic environments
    - Does this mean data augmentation not required for ASVspoof5?
**II.II Database Partition**
- Training Set
    - 400 speakers
    - 19k real utt, 164k fake utt
    - fakes from 8 TTS/VC algs
- Dev Set
    - Subset 1
        - 398 speakers (designated targets for ASV), 1-3 utt each
        - 17k real utt, 110 fake utt
        - fakes use TTS/VS algs diff from those used for train set
    - Subset 2
        - 387 non target speakers
        - 14k real utt
- Eval Set
    - same structure as dev set
    - 367 target speakers, 370 non target speakers
    - real utt: 71k for target, 68k for non target
    - 542k fakes gen using distinct algs not used in other sets
    - All eval data treated with various lossy encoding

- Questions:
    - what kind of noise/artifacts exist in training and dev data?
    - only codecs applied to eval? Why?



---


## Summary

**Architecture**
- E2E raw-waveform models consistently outperform hand-crafted/fixed-feature front ends (Li survey III.I.IV; Müller V.II) — supports the sinc-free 1D residual conv front-end decision.
- Sinc-layer/SincNet remains common even in "E2E" work (RawNet2, AASIST, RawGAT-ST) but isn't clearly necessary — several E2E variants use plain multi-layer 1D conv instead (Li survey III.III), consistent with dropping it.
- GAT-based models (AASIST, RawGAT-ST) top the leaderboards in both Li's survey and Müller's cross-model comparison — the architecture to beat, not necessarily to copy.
- Best-performing clf designs tend to combine local + global pattern extraction (CNN/ResNet + GNN or Transformer ensembles) — relevant to the temporal aggregation decision (self-attention vs. GAT vs. Mamba).
- Fixed 4s input length is standard across AASIST/RawNet2/Müller, but Müller found fixed-length severely degrades performance relative to variable-length input — worth flagging as a tension with the "standard 64,600-sample input" decision made for citability.

**Data Augmentation**
- Noise + codec augmentation showed the largest measured benefit in Li's survey (Table 5), but effectiveness is dataset- and feature-dependent — no universal recipe.
- RawBoost improves reliability on ASVspoof2021 LA specifically (Li survey III.IV.I) — supports treating it as an ablation candidate rather than a default inclusion, especially since ASVspoof 5's more realistic MLS-based conditions may reduce its marginal value (open question, not yet resolved in current reading).
- Combining multiple DA techniques is common practice; over-augmenting (e.g. SpecMix on 100% of data) can degrade performance — augmentation intensity itself is a tunable, not just augmentation type.
- ASVspoof 5's eval set applies lossy encoding uniformly but train/dev partition treatment is unclear from the database paper — needs resolution before finalizing the augmentation plan (open question noted in section 5).

**Loss Function**
- OC-Softmax specifically called out (Li survey III.IV.II) as highly effective for generalizing to unseen attack types while resisting overfitting to known ones — directly relevant given ASVspoof 5's generalization gap problem and already in the planned ablation set.
- Margin-based losses (AM-Softmax, OC-Softmax) target inter/intra-class variance directly, unlike plain CE — theoretical motivation for why they might help closing the generalization gap, not just raw EER.
- Hybrid loss functions (e.g., CE + center loss) flagged as promising but underexplored — potential extension to the planned ablation beyond the four candidates already listed.

**Generalization / Robustness**
- Cross-dataset evaluation is a known failure mode: models trained on ASVspoof 2019 fail badly on out-of-domain data (ITW), partly attributed to channel effect mismatch (Li survey IV.V; Müller V.III, 200–1000% degradation).
- Naive multi-dataset cotraining does not guarantee generalization — this tempers any assumption that just adding more training data (or datasets) fixes the ASVspoof 5 generalization gap.
- This reinforces that the project's zero-shot WaveFake evaluation is a meaningful stress test, not a formality — consistent with existing project scope.

**Open questions to resolve before architecture finalization**
- Whether ASVspoof 5's realistic acoustic conditions reduce the marginal value of synthetic noise/reverb augmentation (raised in section 5, not yet answered by current reading — may need the ASVspoof 5 challenge overview paper for challenge-level consensus).
- Whether fixed 4s input length materially hurts performance for this project's architecture, given Müller's finding — worth a quick empirical check rather than assuming AASIST's convention is optimal.
- GNN internals (stack node, HS-GAL, MGO) still flagged as needing further understanding — noted as non-blocking per earlier discussion, since the project isn't replicating AASIST's graph structure directly.

### Open Question Resolutions

**Data augmentation**
- ASVspoof 5's crowdsourced MLS source data already contains natural acoustic diversity (unlike VCTK-based prior editions), so train/dev don't need synthetic DA to simulate realism. Codecs are applied only to the eval set, as a robustness stress test.
- Practitioners still use DA (RawBoost, MUSAN/RIR, codec aug) to close the train/eval mismatch, but results are mixed — no consensus.
- **Decision:** treat DA as an ablation axis (none / RawBoost / codec / noise+reverb).

**Fixed 4s input length**
- Müller found fixed-length input hurts performance vs. variable-length; ASVspoof 5's longer avg. utterance (~10s vs. 3-5s) sharpens this concern. Standard practice uses a random 4s crop per epoch, not a static one.
- **Decision:** keep 4s as primary for citability against AASIST-L/RawNet2

---

**Further consideration suggests**
- most building most capable front end possible (SSL FE dropped in AASIST to replace from-scratch RawNet2-like FE dramatically improves EER)
- audio-informed/focused model

### 6. LEAF: A LEARNABLE FRONTEND FOR AUDIO CLASSIFICATION -  https://arxiv.org/pdf/2101.08596 - Zeghidour et al 2021

**I Introduction**
- Mel filterbanks standard in audio processing
- perceptually motivated and convenient for ML but pose limitations
- Audio FE 3 stages: filtering -> pooling -> compression/norm
- Other learnable FEs exist but LEAF outperforms w just a few hundred params
- LEAF generalizable to multiple audio tasks (models retrained and tested)
- Propose an improvement to SincNet
- 

**II Related Work**
- Early alternatives to mel filtbanks are heavy (multi layer) in comparison: lightweight alternative motivated here

**II.I Learning from Waveforms**
- Prev learnable filtbanks use: Sinc (BPF), Gammatone init, Scatter Transform (?), Gabor filt
- Authors parameterize learnable complex valued Gabor filts
    - no win func necessary
    - squared mod brings signal back to real value while also performing hilbert xform (envelope extraction)
- Unconstrained filters lead to overfitting + stability issues (authors solve this problem)

**II.II Learning compression / normalization**
- Not as much lit here
- PCEN - per ch energy norm
    - orig for keyword spotting
    - outperforms log compression
    - used in ASR and bioacoustics
- proposed system is PCEN extension (learnable along with other stages)

**III Model**
- Filterbank & nonlin -> pooling (decimation) -> nonlin compression (dynamic range reduction) 
- FE is in context of a wider clf model, so FE params and clf params learned E2E simultaneously

**III.I Filtering**
- Filters complex valued, length of W (is this learnabled param?)
- Two methods: fully param conv filters, learnable Gabor filters
- Using img processing terminology: "stride of 1", might mean convolution is actually correlation?

**III.I.I Normalized 1D Conv**
- First ver of filtering component: standard 1D conv init w bank of Gabor filters (approx mel filtbank)
- Limitations:
    - learn freq selection but also scaling (thus l2 norm applied to coefs before conv)
    - High DOF lead to overfitting
- Some level of parameterization can alleviate these issues (constrained parameterization)
    - Maintain smoothness and interpretability

**III.I.II Gabor 1D convolution**
- Gabor = gaussian modulated by sinusoid
- Pros:
    - optimal tradeoff between time and freq localization
    - interpretable since filters follow a function
    - Quasi-analytic (freq reponse near 0 at negative freq?)
    - Squared modulus brings back to real, env extraction mentioned earlier
- Parameterization
    - center freq and bandwidth (freq response is gaussian): both constrained
    - seems like win length W fixed (25 ms at 16k = 401 samp)
    - brings down param by 200x compared to fully parameterized (W * N vs 2 * N)

**III.I.III Time Freq Analysis and Learnable Filters**
- Learnable freq ranges can introduce freq axis scattering (no longer ordered freq axis in spectrogram)
- Findings:
    - ordered filters at init tend to stay ordered through training
    - enforcing sorted filters has no effect on performance !!THIS IS BIG!!

**III.II Learnable Lowpass Pooling**
- Output of filtering stage = same temporal res as input signal
- Pooling stage = decimation operation
- Prev work tested max pooling, avg pooling, LPF
    - LPF improved
- In ResNet and DenseNet, maxPool or avgPool layers w fixed LPF help
- This is all sounding very image processing-oriented...
- Proposed method extends prev in 2 ways
    - learnable LPF per channel (LPF implemented depth-wise)
    - filters parameterized as Gaussians (Gabor) with fc=0 and learnable bandwidth
- ?? Why LPF a signal that is the result of a BPF? ??
- !! Idea: for learnable filterbank, if bandwidth is learnable constrain at something narrow, then implement band limited sampling based on this max bandwidth... likely 0 extra parameters needed, and seems like most aggressive way to reduce samples after splitting bands!!

**III.II Learning Per-channel Compression and Normalization**
- Prev work: log compression, 3rd root, 10th root,...
- PCEN
    - parameterized exponential moving avg filter w offset
- This work uses channel-dependent learnabled smoothing (a PCEN param)
    - called sPCEN
- !!LEAF comprises: 1D Gabor bank -> Gaussian LPF pooling -> sPCEN!!
    - so each channel gets the above treatment...
    - this means a learned band-lim signal hits a learned LPF then a smoothing filter?
        - where do we end up decimating? implementation of learned LPF for decimation doesn't say
        - how does exp filt smoothing compress dynamic range? this seems more like another LPF

**IV Experiments**
- LEAF eval on single task clf, multi task clf, multi-label clf on AudioSet
- Compare against similar structure implementation of mel filtbank, time domain filterbank, sincnet
- common backbone, diff FE
    - FE -> conv encoder -> FC clf "head(s)"
    - conv encoder has multi million params: why do we care about how many params FE introduces is we're in the millions overall already?
- Generally, 40 ch filt banks, 25ms filters (400 taps at 16kHz)
- "The learnable pooling is computed over 401 samples with a stride of 160 samples (10 ms at 16 kHz), giving
the same output dimension as mel-filterbanks" Okay so how does this translate into decimation?
- 1s input lengths sampled from full inputs to address variable length - this likely hurt performance in some tasks

**IV.I Single Task Audio Clf**
- On Avg this method outperforms
- !!Importantly, does not outperform SincNet for Speaker Identification!!

**IV.II Multi Task Audio Clf**
- diff head trained for each task? Not sure of difference
- Improves performance on SpeakerID

**IV.III Multi Label on AudioSet**
- LEAF wins, not by much

**IV.IV Analysis of learned filters, pooling, and compression**
- Learned fc's don't really deviate much from Mel center freqs
- Learned pooling filts do deviate from pure guassian init: mostly converge on larger bandwidth
    - diff bandwidths confirm worth of depth-wise pooling (variable filts per chan)
- Learned PCEN coefs all init at same value
    - most learned slow mov avg except high freq
    - root coefs also spread

**IV.V Robustness to Noise**
- LEAF and its components seemingly more robust to noise than other FE components mentioned here

**V Conclusion**
- Authors plan to further remove hand-crafted biases
    - learnable filter length and stride
    - ??what does stride mean here if we're literally convolving filter kernel with input signal??
        - sounds more like CNN image processing "convolution"

**Summary from LEAF paper**

- LEAF replaces all three stages of classical audio feature extraction (filtering, pooling, compression/normalization) with fully learnable counterparts, in contrast to SincNet (learnable filtering only) or fixed mel-filterbanks.
- Key design constraint: parameterize filters (Gabor: center freq + bandwidth) rather than fully free convolutional kernels — cuts parameter count ~200x vs. unconstrained filters and avoids overfitting/instability.
- Decimation happens at the pooling stage via strided depthwise convolution (LPF + subsampling in one operation) — not at compression (PCEN).
- PCEN's smoothing filter performs adaptive gain normalization; actual dynamic-range compression comes from the exponent/root terms applied afterward.
- Learned filters stay frequency-ordered without explicit constraint, and enforcing order has no measurable effect on performance — suggests less hand-imposed structure is needed than assumed.
- LEAF does not universally win: underperforms SincNet specifically on speaker ID in single-task setting, only closes/reverses that gap in multi-task training — frontend performance is task-dependent, not a uniform ranking.
- Robustness to noise reported as a strength relative to other frontends tested.
- Uses 1s fixed input length for variable-length handling — same class of limitation flagged by Müller for fixed-length inputs elsewhere in this project's lit review.

**Addressing Questions from LEAF paper**

- "Stride of 1... convolution is actually correlation?"** — Yes. Standard deep learning "convolution" layers (including here) don't flip the kernel, so they're mathematically cross-correlation. True of virtually all frameworks, not LEAF-specific.
- "Why LPF a signal that is the result of a BPF?"** — Different purpose, not redundant. The Gabor BPF selects a frequency band; the subsequent LPF is a standard anti-aliasing filter applied before temporal decimation of the resulting envelope signal, not a second frequency-selection step.
- "Idea: band-limited sampling based on learnable bandwidth"** — legitimate and distinct from what LEAF does. LEAF's pooling is conventional baseband lowpass + decimation; it does not exploit the bandpass sampling theorem to intentionally alias content, unlike the aggressive sample-reduction idea explored earlier in this project's own design discussion. Worth noting as a genuine potential extension beyond LEAF, not something already implemented here.
- "How does exp filter smoothing compress dynamic range?"** — It doesn't directly; the moving-average smoothing (M(t)) is for adaptive normalization/gain control. Actual compression is applied via the exponent (α) and root (r) terms in the PCEN formula, after normalization.
- "Where do we end up decimating?"** — At the pooling stage. The quoted stride-160-samples detail is the decimation step (implemented jointly with the Gaussian LPF as one strided depthwise conv). PCEN operates on already-decimated frames and does not decimate further.
- "Why care about FE params if conv encoder is millions of params already?"** — Valid in LEAF's own context (frontend is a negligible fraction of total budget there). Far more consequential for this project, where total target param count (~85K) makes frontend parameter economy a first-order design constraint rather than a rounding error.
- "What does stride mean if we're literally convolving?"** — Same meaning as in any strided CNN layer: number of samples skipped between successive kernel evaluations. No difference in terminology between this and standard image-processing convolution usage.


