% This must be in the first 5 lines to tell arXiv to use pdfLaTeX.
\pdfoutput=1

\documentclass[11pt]{article}

% ACL style. Remove [review] for final version.
\usepackage[review]{acl}

% Standard packages
\usepackage{times}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{inconsolata}

% Math
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}

% Tables and figures
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{adjustbox}
\usepackage{multirow}
\usepackage{makecell}
\usepackage{tablefootnote}
\usepackage{float}
\usepackage{subcaption}

% Algorithms
\usepackage{algorithm}
\usepackage{algorithmic}

% Lists, boxes, colors, symbols
\usepackage{enumitem}
\usepackage{tcolorbox}
\usepackage{mdframed}
\usepackage{pifont}
\usepackage{xcolor}
\usepackage{colortbl}

% Cross references.
% ACL style usually loads hyperref already, so do not load hyperref again unless needed.
\usepackage[nameinlink,noabbrev]{cleveref}

% Theorem-like environments
% amsthm provides the proof environment and theorem styles.
% Load after amsmath/mathtools.
\usepackage{amsthm}

\theoremstyle{plain}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}[proposition]{Corollary}

\theoremstyle{remark}
\newtheorem*{remark}{Remark}

% Math macros used in the paper
\newcommand{\X}{\mathcal{X}}
\newcommand{\Tset}{\mathcal{T}}
\newcommand{\Dset}{\mathcal{D}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Prb}{\mathbb{P}}
\newcommand{\Ind}{\mathbb{I}}

\DeclareMathOperator*{\argmaxop}{arg\,max}

% Check/cross marks
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}

% Optional author comments
\newcommand{\todo}[1]{{\textcolor{red}{TODO: #1}}}

\definecolor{tablegray}{rgb}{0.9,0.9,0.9}
\definecolor{darkgreen}{rgb}{0,0.5,0}
\definecolor{lightblue}{rgb}{0.7,0.9,1}
\definecolor{lblue}{rgb}{0.25,0.5,1}

\usepackage[normalem]{ulem}

\newcommand{\VH}[1]{{\textcolor{green!100!yellow!70!black!100!}{VH: #1}}}
\def\OD#1{{\color{red}OD: \it #1}}
\def\ODdel#1{\bgroup\markoverwith{\textcolor{red}{\rule[0.4ex]{2pt}{3pt}}}\ULon{#1}}
\def\VDdel#1{\bgroup\markoverwith{\textcolor{green!90!red!55}{\rule[0.4ex]{2pt}{3pt}}}\ULon{#1}}
\usepackage{fontawesome}
% If the title and author information does not fit in the area allocated, uncomment the following
%
%\setlength\titlebox{<dim>}
%
% and set <dim> to something 5cm or larger.

\title{Causal Abstention for Cost-Aware Language-Agent Collaboration}


\begin{document}
\maketitle
\begin{abstract}
Multi-agent language systems are often deployed under the assumption that collaboration improves reasoning quality, but this assumption breaks down once token cost, latency, and collaboration-induced errors enter the picture.
We study protocol selection under a cost-aware utility objective, where the system decides on each input whether to run a collaborative protocol or fall back to solo reasoning.
We propose \textbf{C}ausal \textbf{A}dvantage \textbf{R}outing for \textbf{L}anguage agents (CARL), a contextual-bandit framework that estimates the conditional collaboration advantage of each protocol over solo reasoning.
CARL combines confounded execution logs with budgeted randomized probes, forms one-vs-solo doubly robust targets, and at deployment uses a calibrated lower-confidence-bound policy that invokes a protocol only when its estimated advantage remains positive under uncertainty, otherwise abstaining to solo execution.
Across math, code, and multi-hop question-answering benchmarks with an open-weight model, CARL improves cost-aware utility and substantially reduces harmful collaboration over fixed protocols, compute-matched solo strategies, MAS routers, cascade methods, and causal routing baselines, indicating that effective multi-agent systems should estimate when collaboration is causally worth its cost rather than invoking it by default. Our code and data are available at \url{https://anonymous.4open.science/r/CARL-704E}.
\end{abstract}


\section{Introduction}

Large language models are increasingly embedded in multi-agent systems where specialized instances debate, critique, verify, refine, or decompose problems before producing a final answer \cite{du2023improving,li2023camel,wu2023autogen,madaan2023selfrefine,shinn2023reflexion}. These systems open up multiple reasoning paths and can improve performance on difficult math, coding, and planning tasks \cite{wei2022chain,wang2022selfconsistency,zhou2022least,yao2023tree,li2024moreagents,wang2024mixtureagents}. The approach fits a broader inference-time compute paradigm that invests additional calls at test time rather than retraining a stronger base model \cite{snell2024scaling}.

Collaboration is not free. Each additional turn consumes tokens, adds latency, and opens new failure modes. A verifier can be confidently wrong, a debate can converge on a persuasive but incorrect majority, a critic can overwrite a correct answer, and a workflow can burn through its budget restating uncertainty rather than resolving it. Prior work has also observed that multi-agent gains are heterogeneous and that debate-style procedures can be unreliable in some settings \cite{liang2023divergent,multiagentbench2025}. In production, the relevant objective is not raw accuracy but cost-aware utility, the reward from a better answer minus the monetary and latency cost of the collaboration used to obtain it.

Existing approaches only partially address this objective. MAS routers such as MasRouter \cite{yue2025masrouter} learn to select among collaboration modes and role configurations, but they are framed as predictive routing methods rather than as estimators of a causal one-vs-solo advantage. Cascade methods such as CascadeDebate \cite{chang2026cascadedebate} insert multi-agent deliberation at confidence-based escalation boundaries, yet they trigger collaboration on difficulty or confidence signals rather than on whether the specific protocol is causally beneficial for the current input. Cost-aware model routers such as FrugalGPT and RouteLLM \cite{chen2023frugalgpt,routellm2024,ding2024hybridllm,srivatsa2024harnessing} together with causal LLM routing from observational data \cite{tsiourvas2025causal}, route among models rather than collaborative protocols. None of these directly target the counterfactual quantity at the heart of selective collaboration: for a given input, the cost-aware utility difference between running a protocol $T_k$ and running solo reasoning.

We refer to this quantity as the conditional collaboration advantage, and it serves as the central estimand for protocol selection in our framework. Estimating it from probe-supported logs fits the doubly robust framework \cite{robins1994estimation,dudik2011doubly,chernozhukov2018double}, whose pseudo-outcomes remain consistent when either the outcome or the propensity model is correctly specified. The same estimand directly supports abstention, since the system can fall back to solo execution whenever no collaborative treatment shows a positive estimated advantage. Building on this view, we propose \textbf{C}ausal \textbf{A}dvantage \textbf{R}outing for \textbf{L}anguage agents (CARL). During training, CARL combines confounded execution logs with budgeted randomized probes and forms one-vs-solo doubly robust advantage targets. At deployment, a calibrated lower-confidence-bound (LCB) policy invokes the protocol with the highest positive pessimistic advantage and falls back to solo otherwise, linking selective collaboration to calibrated abstention from selective prediction and uncertainty quantification \cite{chow1970optimum,vovk2005algorithmic,angelopoulos2021gentle}.

Our contributions are as follows.
\begin{itemize}[leftmargin=14pt]
  \item We formulate protocol selection around the collaboration advantage, the causal treatment effect of a collaborative protocol over solo execution, and cast it as a contextual-bandit problem. This estimand separates CARL from MAS routers that optimize predictive quality and from cascade methods that trigger on difficulty or confidence signals.
  \item Building on this formulation, we develop a cost-decomposed doubly robust advantage estimator together with a calibrated pessimistic LCB deployment policy. The harm bound in Proposition~\ref{prop:harm} acts as a conditional calibration diagnostic rather than a distribution-free safety guarantee, and we report empirical harmful-collaboration rate and selected-action lower-bound coverage as the corresponding metrics.
  \item On math, code, and multi-hop QA benchmarks with an open-weight model, CARL improves cost-aware utility and reduces harmful collaboration over fixed protocols, compute-matched solo strategies, MasRouter-style routing, CascadeDebate-style selective deliberation, causal routing baselines, and causal ablations including DM, IPW, DR-Greedy, and DR-LCB.
\end{itemize}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{latex/CARL.pdf}
  \caption{Overview of the CARL framework, which estimates protocol-specific collaboration advantages from probe-supported logs and deploys a calibrated LCB policy to decide when to invoke multi-agent collaboration or abstain to solo reasoning.}
  \label{fig:CARL_overview}
\end{figure*}

\section{Related Work}
% =========================================================

\paragraph{Multi-agent LLM systems and selective collaboration.}
Collaborative LLM systems include role-based societies, debate, 
critique-and-revision, verifier-augmented generation, and agent 
frameworks 
\cite{park2023generative,qian2023chatdev,hong2023metagpt,chen2023agentverse,chan2023chateval,khan2024debating,cobbe2021training,lightman2023verify}. 
Gains are input-dependent, and debate can amplify errors 
\cite{cemri2025why,gao2025single}. MasRouter routes among MAS 
configurations; CascadeDebate escalates to deliberation in a cascade. 
CARL instead targets the causal advantage over solo and treats 
abstention as a first-class action.

\paragraph{Cost-aware routing, causal routing, and inference-time compute.}
FrugalGPT and RouteLLM study cost-aware routing among models 
\cite{chen2023frugalgpt,routellm2024}. Causal LLM Routing learns 
routing policies from observational data with causal objectives. 
Their action is a model. CARL's action is a full protocol (roles, 
prompts, topology, stopping rule, aggregation), shifting the 
estimand to one-vs-solo advantage net of cost.

\paragraph{Abstention, calibration, and doubly robust estimation.}
Abstention originates in selective classification 
\cite{elyaniv2010foundations,geifman2017selective}. Conformal and 
calibration methods support uncertainty-aware prediction 
\cite{romano2019conformalized,angelopoulos2022conformalrisk}. Recent 
LLM work studies abstention and refusal calibration 
\cite{yin2023large,xiong2023llms,chen2023adaptationselfevaluationimproveselective,kirichenko2025abstentionbenchreasoningllmsfail}. 
CARL abstains from \emph{collaboration}, not from answering. The 
estimator adapts doubly robust methods 
\cite{jiang2016doubly,thomas2016data} to one-vs-solo advantage with 
explicit token and latency cost.


\section{Methodology}
\label{sec:method}

\subsection{Setup and the Collaboration Advantage}

Let $X\in\X$ denote the pre-treatment context of one episode. In our system $X$ records input descriptors (token length, task type, benchmark identifier), retrieval features for the code task (BM25 score and top-$k$ document length statistics), externally measured difficulty signals (problem-class labels and task-source indicators), and the entropy of a routing-only cheap solo probe whose generated text is discarded and never used by any downstream protocol. All entries of $X$ are computed before the routing decision. Solver messages, verifier critiques, debate transcripts, and judge rationales are post-treatment mediators and excluded from $X$. The full feature list and the routing signals we deliberately exclude are in \cref{app:features}, and \cref{app:sensitivity} reports a sensitivity analysis for the residual risk that a logged routing signal was not recorded.

The action space $\Tset=\{T_0,\ldots,T_K\}$ is a finite catalog of fully specified protocols, where each $T_k$ fixes the model checkpoint, prompts, roles, decoding parameters, communication topology, turn budget, stopping rule, and aggregation rule. The main experiments use five protocols. $T_0$ (\texttt{SOLO}) is a one-agent direct answer and serves as the abstention baseline. $T_1$ (\texttt{SELF\_REFLECT}) keeps the same model and runs a second critique-and-revision turn. $T_2$ (\texttt{VERIFY}) introduces an independent verifier that checks the solver's answer over two turns. $T_3$ (\texttt{DEBATE\_2}) runs two proposers in a two-to-three-turn debate adjudicated by a judge, and $T_4$ (\texttt{PROPOSE\_VERIFY}) generates multiple proposals and then verifies them across three turns. Per-protocol token counts are task-dependent, with BM25 file retrieval adding roughly 8K input tokens per call on SWE-bench Lite; \cref{tab:compute_cost} gives the breakdown.

For protocol $T_k$ the cost-aware potential outcome is
\begin{equation}
Y(k)=R(k)-\lambda C_{\mathrm{tok}}(k)-\mu C_{\mathrm{lat}}(k),
\label{eq:outcome}
\end{equation}
where $R(k)\in[0,1]$ is a task reward (exact match for MATH-500, the resolved indicator from the official execution-based harness for the code task, F1 for QA), $C_{\mathrm{tok}}(k)$ is the total number of input plus output tokens divided by a per-task reference budget, and $C_{\mathrm{lat}}(k)$ is end-to-end wall-clock latency divided by a per-task reference latency. After normalization, $R$ and the two cost terms all live on roughly the unit scale, which makes $Y$ comparable across tasks. The default operating point is $\lambda=0.05$ and $\mu=0.02$, chosen on a development split so that solo and the cheapest collaborative protocol have comparable mean utility. The $(\lambda,\mu)$ sweep is in \cref{app:lambda_sweep}.

Writing $m_k^\star(x)=\E[Y(k)\mid X=x]$ for the conditional mean utility of protocol $T_k$, CARL targets the conditional collaboration advantage
\begin{equation}
\tau_k(x)=m_k^\star(x)-m_0^\star(x), \qquad k\ge1.
\label{eq:advantage}
\end{equation}
A collaborative protocol is harmful on input $x$ when $\tau_k(x)<0$. The oracle policy collaborates only when the best advantage is positive,
\begin{equation}
\pi^\star(x)=
\begin{cases}
T_{\argmaxop_{k\ge1}\tau_k(x)}, & \text{if } \max_{k\ge1}\tau_k(x)>0,\\
T_0, & \text{otherwise}.
\end{cases}
\label{eq:oracle_policy}
\end{equation}
The one-vs-solo form is important because historical logs are difficulty-biased: expensive protocols are often assigned to harder inputs, so the logged contrast decomposes as
\begin{align}
& \E[Y\mid T{=}T_k]-\E[Y\mid T{=}T_0] \nonumber\\
&\quad = \E[\tau_k(X)\mid T{=}T_k] \nonumber\\
&\qquad + \E[m_0^\star(X)\mid T{=}T_k] \nonumber\\
&\qquad - \E[m_0^\star(X)\mid T{=}T_0],
\label{eq:selection_bias}
\end{align}
where the last two terms form a difficulty-selection bias that can make useful protocols appear harmful. CARL therefore estimates causal advantage over solo rather than predicting which protocol has the highest logged reward.

\subsection{Probe-Supported Logs and Propensities}

Training data are logged episodes $\Dset=\{(X_i,T_i,Y_i,p_i)\}_{i=1}^n$, where $p_i$ is the assignment probability of the executed treatment under the live behavior policy
\begin{equation}
b(T\mid X)=(1-\epsilon_{\mathrm{probe}})\pi_\phi(T\mid X)+\epsilon_{\mathrm{probe}}\,q_\psi(T\mid X).
\label{eq:behavior_policy}
\end{equation}
We use $\epsilon_{\mathrm{probe}}=0.15$. The randomized probe component $q_\psi$ is the identification anchor and depends only on $X$. It is a softmax of epistemic uncertainty, an inverse-count coverage bonus, and a small negative cost term. We enforce overlap directly at the behavior-policy level by requiring $b(T_k\mid x)\ge b_{\min}=0.02$ for every $x$ and $k$, so the propensity clip $e_{\min}=0.02$ used at training time is consistent with the deployed exploration floor. The cost penalty inside $q_\psi$ shifts probability mass within the admissible region without violating this floor, so expensive protocols are not starved of support on hard inputs, where the causal contrast is most informative.

Three sources of treatment assignment appear in $\Dset$, and we keep them separate. For randomized probe rows the propensity is exact. For exploitation rows the propensity is logged from the deployed $\pi_\phi$. Legacy observational logs whose propensities are unknown are used only for descriptive analysis and never enter the DR construction, which removes the unknown-propensity case from the causal pipeline. The cross-fitted propensity model serves only as a fallback when a logged probability is missing for a recoverable reason (details in \cref{app:training}). Clipping introduces a bounded bias that we treat as part of the bias-variance tradeoff and audit through the diagnostics in \cref{sec:experiments}.

\subsection{Cost-Decomposed Outcome Models and DR Advantage Scores}

A shared encoder $h_\theta(X)$ feeds treatment-specific reward, 
token-cost, and latency heads, and the utility prediction recombines 
them as
\begin{equation}
\hat m_k(x)=\hat r_k(x)-\lambda \hat c^{\mathrm{tok}}_k(x)-\mu \hat c^{\mathrm{lat}}_k(x).
\label{eq:cost_decomp}
\end{equation}
The three heads are trained on their natural targets ($R_i$, 
normalized token cost, normalized latency) rather than on the 
composite $Y_i$. This decomposition partitions the pipeline into a 
cost-independent stage (the shared encoder and the reward, 
token-cost, and latency heads, trained once) and a 
cost-dependent stage (the advantage head and the conformal quantile, 
which depend on $(\lambda,\mu)$ through $Y$). Changing the deployment 
cost coefficients to some new $(\lambda',\mu')$ therefore requires 
refitting only the advantage head and recalibrating $a_k$ from the 
cached nuisance predictions, which is a small fraction of total 
training cost (\cref{tab:compute_cost}). All nuisances are 
cross-fitted across five folds.

For each non-solo protocol $T_k$, the one-vs-solo doubly robust score is
\begin{align}
\phi_{i,k} &= \hat m_k(X_i)-\hat m_0(X_i) \nonumber\\
& \quad + \frac{\Ind[T_i{=}T_k]}{\hat e_k(X_i)}\bigl(Y_i-\hat m_k(X_i)\bigr) \nonumber\\
& \quad - \frac{\Ind[T_i{=}T_0]}{\hat e_0(X_i)}\bigl(Y_i-\hat m_0(X_i)\bigr).
\label{eq:dr}
\end{align}

\begin{proposition}[One-vs-solo doubly robust score]
\label{prop:dr}
With $e_k(x)=\Prb(T{=}T_k\mid X{=}x)$, and under consistency, overlap, 
and conditional exchangeability given $X$,
\begin{align}
& \E[\phi_{i,k}\mid X_i{=}x]-\tau_k(x) \nonumber\\
&\ = \Bigl(1-\tfrac{e_k(x)}{\hat e_k(x)}\Bigr)\bigl(\hat m_k(x)-m_k^\star(x)\bigr) \nonumber\\
&\ \ - \Bigl(1-\tfrac{e_0(x)}{\hat e_0(x)}\Bigr)\bigl(\hat m_0(x)-m_0^\star(x)\bigr).
\label{eq:dr_bias}
\end{align}
So $\E[\phi_{i,k}\mid X_i{=}x]=\tau_k(x)$ whenever the relevant 
outcome models or the relevant propensities are correct.
\end{proposition}

The score $\phi_{i,k}$ is a standard DR object and inherits the 
identification properties in \cref{prop:dr}. Its form is fixed in 
advance by three choices specific to selective collaboration: the 
action is a full protocol rather than a model, the contrast is taken 
against solo rather than between arbitrary arms, and the residuals 
reuse the cost-decomposed heads so that downstream LCB pessimism 
applies to advantage rather than absolute utility.

To stabilize the DR target under sparse local support we shrink it 
toward the direct contrast,
\begin{equation}
\tilde\tau_{i,k}=\rho_{i,k}\,\phi_{i,k}+(1-\rho_{i,k})\bigl(\hat m_k(X_i)-\hat m_0(X_i)\bigr),
\label{eq:shrinkage}
\end{equation}
\begin{equation}
\rho_{i,k}=\frac{n_{\mathrm{eff}}(X_i,T_k)}{n_{\mathrm{eff}}(X_i,T_k)+\gamma\,\hat v_{i,k}},
\label{eq:rho}
\end{equation}
where $n_{\mathrm{eff}}(X_i,T_k)$ is a $K_{\mathrm{NN}}$-nearest-neighbor 
effective sample size for treatment $T_k$ in feature space, 
$\hat v_{i,k}$ is the bootstrap variance of $\phi_{i,k}$ across 
cross-fitting folds, and $\gamma$ is selected on the validation split. 
The shrinkage is a stabilization step rather than a DR-preserving 
transformation: when $\rho_{i,k}<1$, $\tilde\tau_{i,k}$ is a convex 
combination of $\phi_{i,k}$ and the direct-model contrast, trading 
a controlled bias toward outcome-model assumptions for lower 
variance under sparse local support. \cref{prop:dr} therefore applies 
to $\phi_{i,k}$, not to $\tilde\tau_{i,k}$, which we treat as a 
stabilized DR-like target; its behavior under model and propensity 
misspecification is audited through the OPE diagnostics in 
\cref{tab:ope} and the calibration metrics in 
\cref{tab:calibration_combined}. The advantage learner then regresses 
$\tilde\tau_{i,k}$ on $X_i$, producing an advantage estimate 
$\tilde\tau_k(x)$ and an epistemic uncertainty estimate 
$\hat\sigma^{(\tau)}_k(x)$ defined as the standard deviation of a 
five-member ensemble of advantage heads trained on bootstrap 
resamples of $\{\tilde\tau_{i,k}\}$, plus a residual noise floor 
calibrated on validation probes. The same construction is used for 
every protocol, which keeps $\hat\sigma^{(\tau)}_k$ comparable across 
treatments.

\begin{table*}[!tp]
\centering
\small
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular*}{\textwidth}{l@{\extracolsep{\fill}}cccccccccc}
\toprule
& \multicolumn{3}{c}{\textbf{Math (MATH-500)}}
& \multicolumn{3}{c}{\textbf{Code (SWE Lite)}}
& \multicolumn{3}{c}{\textbf{QA (MuSiQue+2Wiki)}}
& \textbf{Avg.} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-11}
Method & Acc & Util & CHR & Res. & Util & CHR & F1 & Util & CHR & Util \\
\midrule
\multicolumn{11}{l}{\textit{Group 1: Fixed protocols and compute-matched solo}} \\
Always-SOLO              & .54 & .51 & N/A & .28 & .25 & N/A & .49 & .47 & N/A & .41 \\
Always-VERIFY            & .55 & .55 & .30 & .30 & .27 & .50 & .50 & .49 & .40 & .44 \\
Always-DEBATE            & .56 & .53 & .45 & .31 & .26 & .65 & .51 & .43 & .55 & .41 \\
Best-Fixed               & .55 & .55 & .30 & .30 & .27 & .50 & .50 & .48 & .36 & .43 \\
Solo-SelfConsist.        & .60 & .55 & N/A & .32 & .28 & N/A & .55 & .51 & N/A & .45 \\
Solo-Rerank              & .61 & .56 & N/A & .33 & .29 & N/A & .55 & .52 & N/A & .46 \\
\midrule
\multicolumn{11}{l}{\textit{Group 2: Predictive MAS routing and selective collaboration}} \\
MasRouter-style          & .60 & .55 & .24 & .34 & .30 & .30 & .55 & .51 & .26 & .45 \\
MasRouter-style+Solo     & .61 & .57 & .17 & .36 & .32 & .22 & .56 & .53 & .19 & .47 \\
CascadeDebate-style      & .60 & .56 & .19 & .35 & .31 & .24 & .56 & .52 & .22 & .46 \\
Adaptive-Orch.+Abs.      & .61 & .57 & .18 & .35 & .31 & .23 & .56 & .52 & .20 & .47 \\
\midrule
\multicolumn{11}{l}{\textit{Group 3: Causal / offline bandit components}} \\
DM-Greedy                & .58 & .54 & .21 & .34 & .30 & .25 & .53 & .49 & .23 & .44 \\
DR-Greedy                & .61 & .57 & .16 & .36 & .32 & .21 & .56 & .52 & .18 & .47 \\
DR-LCB (absolute util.)  & .62 & .58 & .14 & .37 & .33 & .17 & .57 & .53 & .16 & .48 \\
Outcome-Reg.+LCB         & .60 & .57 & .13 & .36 & .32 & .19 & .56 & .52 & .17 & .47 \\
\midrule
CARL                     & \textbf{.64} & \textbf{.61} & \textbf{.08} & \textbf{.39} & \textbf{.35} & \textbf{.11} & \textbf{.59} & \textbf{.55} & \textbf{.11} & \textbf{.50} \\
Catalog Oracle           & .71 & .68 & .00 & .49 & .44 & .00 & .65 & .60 & .00 & .57 \\
\bottomrule
\end{tabular*}
\caption{Main results, averaged over five seeds. Best-Fixed uses VERIFY/VERIFY/SELF\_REFLECT for Math/Code/QA. Catalog Oracle is the in-catalog oracle, not a benchmark upper bound. CHR is N/A for solo-only methods. For Code, Res.\ and Util.\ are computed on the full 300-instance Lite test split, while CHR and the Catalog Oracle row are computed on the 100-instance audit subset.}
\label{tab:main}
\end{table*}

\subsection{Calibrated LCB Abstention}

A standard router selects the protocol with the highest predicted utility. CARL instead places uncertainty on the abstention boundary $\tau_k(x)=0$, since collaboration should be invoked only when the extra protocol is likely to improve over solo after accounting for cost.

For calibration we hold out 1{,}500 paired-execution instances each for Math and QA, and 300 for Code (limited by SWE-bench Lite's size and drawn from the train set; see \cref{tab:dataset_splits}), on which solo and each candidate protocol are each executed once under deterministic decoding, or averaged over three executions when deterministic decoding is unavailable. These paired executions yield empirical advantages $\tau^{\mathrm{eval}}_{i,k}$ that are noisy estimates of the true protocol advantage. In lower-budget settings, this paired matrix can be replaced by randomized audit probes at the cost of wider lower bounds. The compute and dollar cost of building the calibration matrix is reported in \cref{tab:compute_cost}, and the cost is amortized over deployment because calibration only needs to be redone when the protocol catalog or the base model changes.

On the calibration split we form one-sided standardized residuals
\begin{equation}
s_{i,k}=\frac{\tilde\tau_k(X_i)-\tau^{\mathrm{eval}}_{i,k}}{\hat\sigma^{(\tau)}_k(X_i)+10^{-6}},
\label{eq:cal_resid}
\end{equation}
and set $a_k=\mathrm{Quantile}_{1-\delta}\bigl(\{s_{i,k}\}\bigr)$. The deployment lower bound is $\mathrm{LCB}_k(x)=\tilde\tau_k(x)-\kappa\, a_k\,\hat\sigma^{(\tau)}_k(x)$, and the policy is
\begin{equation}
\hat T(x)=
\begin{cases}
T_{\hat k}, & \text{if } \max_{k\ge1}\mathrm{LCB}_k(x)>0,\\
T_0, & \text{otherwise,}
\end{cases}
\label{eq:lcb_policy}
\end{equation}
with $\hat k=\argmaxop_{k\ge1}\mathrm{LCB}_k(x)$. The main experiments use $\delta=0.1$ and $\kappa=1.0$. We also report selected-action coverage, defined as the fraction of test instances on which the selected non-solo action satisfies $\tau_{\hat T(X)}(X)\ge \mathrm{LCB}_{\hat T(X)}(X)$, with solo treated as covered by construction. The following diagnostic ties lower-bound calibration to harmful collaboration.

\begin{proposition}[Calibration-dependent harm diagnostic]
\label{prop:harm}
If $\Prb(\tau_k(X)\ge \mathrm{LCB}_k(X))\ge 1-\delta$ for every $k\ge1$, then
\begin{equation}
\Prb\bigl(\tau_{\hat T(X)}(X)<0,\ \hat T(X)\ne T_0\bigr)\le K\delta.
\label{eq:harm_bound}
\end{equation}
\end{proposition}

This is a diagnostic rather than a distribution-free safety guarantee, and the bound is loose when $K$ is nontrivial. We therefore report empirical harmful-collaboration rate and selected-action coverage in \cref{sec:experiments} alongside utility.


\subsection{Training and Inference Summary}

Training proceeds in three passes. The first pass collects logs by sampling $T_i\sim b(\cdot\mid X_i)$, executing the chosen protocol, and recording $Y_i$, token count, latency, propensity, and a treatment hash. The second pass cross-fits the reward, token-cost, latency, and propensity nuisances, builds the shrunken DR targets in \cref{eq:dr,eq:shrinkage}, and trains the advantage and uncertainty heads. The third pass executes the paired calibration matrix and fits $a_k$ from \cref{eq:cal_resid}. At deployment, CARL applies \cref{eq:lcb_policy} and falls back to $T_0$ whenever every non-solo lower bound is non-positive. Encoder architecture, loss functions, the number of folds and bootstrap replicates, hyperparameter ranges, and the handling of stochastic decoding are reported in \cref{app:training}.


\section{Experiments}
\label{sec:experiments}
% =========================================================




\subsection{Setup}
\label{sec:setup_exp}

\paragraph{Model and benchmarks.}
All main experiments use \texttt{Qwen/Qwen3.5-27B} under constrained
single-pass protocols: fixed prompts, BM25 context for Code, no
interactive repository editing or test-feedback repair, and fixed
decoding and serving settings. We evaluate on three task families:
MATH-500 (Math, exact-match); SWE-bench Lite with the official
execution-based harness (Code, resolved indicator on the 300 Lite
test instances); and MuSiQue-Ans plus 2WikiMultiHopQA stratified by
hop count (QA, F1). Each task has five splits (\cref{tab:dataset_splits});
Math and QA replay the test set under all protocols, while Code runs
full-matrix diagnostics on a stratified 100-instance audit subset.
Code numbers should be read against this constrained setting rather
than agent-scaffolded reports; cross-model runs are exploratory; no
hyperparameter or calibration quantile is tuned on test or audit data.

\paragraph{Baselines.}
We group baselines into fixed protocols and compute-matched solo
(Group 1), predictive MAS routers and selective-deliberation methods
(Group 2), and causal or offline-bandit ablations (Group 3). The
+Solo variants give a router the same abstention action as CARL.
Full definitions and additional baselines (Confidence-Trigger,
Causal-Routing-style, Naive-Obs.) are in \cref{app:baseline_details}.

\paragraph{Metrics.}
The primary metric is cost-aware utility $Y$, with task-native raw
reward (EM / resolved / F1) as a secondary signal. Harmful-collaboration
rate (CHR) is the fraction of test inputs on which the policy invokes
a collaborative protocol with negative realized advantage over solo;
positive-treatment correctness (PTC) is the complementary fraction
among invoked collaborative actions. Selected-action coverage is the
fraction of test instances where the selected non-solo action satisfies
$\tau_{\hat T(X)}(X)\ge \mathrm{LCB}_{\hat T(X)}(X)$. Means are over five
seeds; per-seed numbers and bootstrap SE are in \cref{app:extra_results}.



\begin{table}[!tp]
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{llcccc}
\toprule
Task & Est. & SR & VER & DEB & PV \\
\midrule
\multirow{2}{*}{Math}
 & Log   & $-.012$ & $-.071$ & $-.029$ & $-.046$ \\
 & Audit & $+.024$ & $+.038$ & $+.018$ & $+.027$ \\
\midrule
\multirow{2}{*}{Code}
 & Log   & $+.003$ & $-.054$ & $-.041$ & $-.061$ \\
 & Audit & $+.011$ & $+.022$ & $+.008$ & $+.014$ \\
\midrule
\multirow{2}{*}{QA}
 & Log   & $+.019$ & $+.008$ & $-.058$ & $-.030$ \\
 & Audit & $+.013$ & $+.021$ & $-.036$ & $-.014$ \\
\bottomrule
\end{tabular}
\caption{Logged contrast vs.\ audit-matrix contrast of each non-solo protocol's advantage over solo. Sign reversals on VER (Math, Code) are consistent with difficulty-based confounding.}
\label{tab:confounding}
\end{table}

\subsection{Main Results}

\paragraph{Overall result.}
\Cref{tab:main} reports the main result. CARL achieves the highest non-oracle cost-aware utility on all three task families, with asymmetric gains across tasks. On Math, CARL improves utility by 0.03 over DR-LCB and 0.04 over MasRouter-style+Solo, and its CHR is 0.08 compared with 0.14 for DR-LCB and 0.17 for MasRouter-style+Solo. On QA, the utility gap to the strongest learned baseline is smaller (about 0.02), and the dominant gain is the CHR drop from 0.16 to 0.11. On Code, the resolved-rate improvement over MasRouter-style+Solo is about three points, with a comparable utility gain after accounting for patch-generation cost, and the CHR on the audit subset drops from 0.22 to 0.11.

\paragraph{Why fixed and predictive baselines fall behind.}
These gaps are driven by substantial input-level heterogeneity: on the full-matrix splits, every collaborative protocol is harmful on 46\% of inputs on average, and oracle-optimal on the remaining 54\%, with verification harmful on roughly one third of math instances and debate harmful on more than half of code audit instances. CARL's selected-action mix tracks this pattern—solo dominates on Code (51\%) and QA (55\%), while verification carries most of the non-solo mass on Math. Per-task and per-protocol breakdowns are in \cref{app:extra_results}.

\paragraph{Targeted comparisons.}
Three comparisons are worth singling out. Adding the solo abstention action to a predictive router (MasRouter-style $\to$ MasRouter-style+Solo) accounts for most of the CHR improvement on QA and Code, but the residual gap to CARL on both utility and CHR remains, suggesting that the predictive objective does not internalize the cost of collaborating on inputs where solo is already adequate. CascadeDebate-style triggers deliberation when solo confidence is low, which is helpful on QA but less so on Code, where confidence on hard issues is a poor proxy for whether a multi-agent patch will resolve the test. DR-LCB places an LCB on absolute utility rather than on the one-vs-solo advantage. It can still select solo, but its conservative boundary is not aligned with the incremental value of collaboration, so it may collaborate when a non-solo protocol has high absolute utility but little advantage over solo.

\paragraph{Catalog-oracle recovery.}
Using per-task catalog-oracle recovery $(U(\pi)-U(\text{Solo}))/(U(\text{Catalog Oracle})-U(\text{Solo}))$, CARL recovers 59\% of the catalog-oracle improvement on Math, 53\% on Code, and 62\% on QA. The recovery is smallest on Code, where collaboration is most often harmful and the catalog-oracle gap on the audit subset is widest. Paired seed-level comparisons against the strongest learned baselines are in \cref{app:extra_results}.

\begin{table}[!tp]
\centering
\small
\setlength{\tabcolsep}{10pt}
\begin{tabular}{lcccc}
\toprule
Task & DM & IPW & SNIPW & DR \\
\midrule
Math  & .047 & .039 & .033 & \textbf{.021} \\
Code  & .078 & .065 & .054 & \textbf{.034} \\
QA    & .053 & .046 & .038 & \textbf{.027} \\
\bottomrule
\end{tabular}
\caption{OPE error against the audit matrix, averaged over five seeds. DR has the lowest error on every task.}
\label{tab:ope}
\end{table}

\subsection{Confounding Diagnostics}

A central motivation for CARL is that observational protocol logs are difficulty-biased, since harder inputs are more likely to be routed to expensive collaborative protocols. We therefore do not interpret logged contrasts as causal effects on their own. Identification relies on randomized probes with logged exact propensities, consistency of the protocol definitions, sufficient overlap after clipping, and the assumption that all routing signals used by the behavior policy are recorded in $X$. Sensitivity analyses for the last assumption are in \cref{app:sensitivity}.

\Cref{tab:confounding} illustrates the difficulty-selection failure mode by comparing the logged observational contrast with the held-out audit-matrix contrast. On Math and Code, verification appears slightly harmful in logged data and reverses sign under the audit-matrix contrast, consistent with harder inputs being preferentially routed to verification. On QA, debate remains harmful under both estimates, indicating average harm in this setting rather than only selection bias. \Cref{tab:ope} reports off-policy evaluation error of the four estimators against the audit matrix. DR has the smallest error on every task, and the gap to IPW and SNIPW is largest on Code, where overlap is the hardest to maintain.

\subsection{Ablations and Abstention Alternatives}

\begin{table}[!tp]
\centering
\small
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lccc}
\toprule
Variant & Util $\uparrow$ & CHR $\downarrow$ & PTC $\uparrow$ \\
\midrule
\textbf{Full CARL}              & \textbf{.58} & \textbf{.10} & \textbf{.81} \\
\midrule
Absolute utility                & .55 & .15 & .75 \\
Single utility head             & .54 & .14 & .77 \\
No DR shaping                   & .53 & .17 & .73 \\
Greedy, no LCB                  & .58 & .22 & .73 \\
Uniform probing                 & .56 & .11 & .80 \\
No shrinkage                    & .56 & .13 & .78 \\
No propensity clip              & .54 & .15 & .75 \\
Shared head                     & .53 & .17 & .73 \\
\bottomrule
\end{tabular}
\caption{Component ablations of CARL, averaged over Math and QA.}
\label{tab:ablation_components}
\end{table}

\begin{table}[!tp]
\centering
\small
\setlength{\tabcolsep}{9pt}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lccc}
\toprule
Method & Util $\uparrow$ & CHR $\downarrow$ & Abst. \\
\midrule
\textbf{CARL}             & \textbf{.58} & \textbf{.10} & .48 \\
\midrule
MasRouter-style+Solo      & .55 & .18 & .30 \\
Outcome-Reg.+LCB          & .55 & .15 & .51 \\
CascadeDebate-style       & .54 & .21 & .46 \\
Solo confidence threshold & .52 & .19 & .44 \\
\bottomrule
\end{tabular}
\caption{Simple abstention alternatives, averaged over Math and QA.}
\label{tab:ablation_abstention}
\end{table}

\Cref{tab:ablation_components} reports component ablations averaged over Math and QA. Replacing the one-vs-solo advantage target with absolute utility closes most of the gap to predictive baselines, confirming that the choice of estimand drives CARL's CHR advantage rather than the DR correction alone. Collapsing the cost-decomposed heads into a single utility head also lowers utility and raises CHR, since the resulting model cannot internalize cost as a learnable structure. Removing DR shaping lowers utility further, consistent with outcome regression alone being insufficient under difficulty-biased logs, and removing LCB pessimism keeps utility high but raises CHR substantially, exposing the utility-harm tradeoff at the abstention boundary. Switching to uniform probing gives up sample efficiency without much CHR change, since overlap is preserved either way.

\Cref{tab:ablation_abstention} contrasts CARL with simple abstention strategies. Calibrated thresholding on solo confidence and confidence-triggered collaboration each reduce some harmful calls but neither matches CARL's tradeoff, since their abstention boundary is not aligned with the one-vs-solo advantage. The $\kappa$ sweep, the cost-decomposition ablation under shifted $(\lambda,\mu)$, and a $b_{\min}$ sensitivity study are in \cref{app:extra_results}.

\subsection{Calibration and Coverage}

\begin{table}[!tp]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccc}
\toprule
Method & Task & ECE $\downarrow$ & PTC $\uparrow$ & Cov. $\uparrow$ \\
\midrule
\multirow{3}{*}{Outcome-Reg.+LCB}
        & Math & 0.10 & 0.76 & 0.86 \\
        & Code & 0.14 & 0.60 & 0.81 \\
        & QA   & 0.11 & 0.61 & 0.84 \\
\midrule
\multirow{3}{*}{DR-LCB (abs.)}
        & Math & 0.08 & 0.80 & 0.88 \\
        & Code & 0.12 & 0.73 & 0.84 \\
        & QA   & 0.09 & 0.72 & 0.87 \\
\midrule
\multirow{3}{*}{CARL}
        & Math & \textbf{0.05} & \textbf{0.86} & \textbf{0.91} \\
        & Code & \textbf{0.08} & \textbf{0.78} & \textbf{0.87} \\
        & QA   & \textbf{0.06} & \textbf{0.76} & \textbf{0.89} \\
\bottomrule
\end{tabular}
\caption{Treatment-effect calibration, PTC, and selected-action coverage. Target coverage 90\%.}
\label{tab:calibration_combined}
\end{table}

\Cref{tab:calibration_combined} reports treatment-effect calibration error, PTC, and selected-action coverage for the LCB-based methods. CARL has the lowest ECE on every task and the highest PTC, and its per-task selected-action coverage is close to the nominal 90\% target. Coverage on Code is slightly lower than on Math and QA, consistent with the larger DR variance on the audit subset and reflected in the wider lower bounds CARL learns on Code.

\subsection{Robustness}

We evaluate three stress settings: injected protocol failures, rate-limit-style missing responses, and adversarial distractor critiques. Treatment-drift stress tests, sequential deployment diagnostics, and an exploratory cross-model robustness study (Llama-3.1-8B, Qwen3.5-27B, Qwen2.5-72B) are in \cref{app:extra_results,app:cross_model}. We do not make a numerical robustness claim in the main text since the relevant tables live in the appendix, and the stress settings should be read as diagnostics for the finite protocol catalog rather than evidence of robustness to arbitrary deployment failures.





\section{Conclusion}

CARL frames protocol selection around the collaboration advantage, defined as the causal treatment effect of each collaborative protocol over solo reasoning. This estimand distinguishes CARL from predictive MAS routers, confidence-triggered cascades, and causal model routers. Across math, code, and multi-hop QA benchmarks, CARL's doubly robust advantage estimation and calibrated LCB abstention improve cost-aware utility and reduce harmful collaboration. The practical implication is that systems should estimate whether collaboration is causally worth its cost on the current input before scaling up agent societies.

\section*{Limitations}

CARL relies on probe-supported data collection, which adds upfront training cost and may be impractical when collaborative protocol executions are expensive on a per-call basis. The causal claims depend on three conditions that we audit but cannot guarantee in arbitrary deployments: the assignment propensities of both probe and exploitation rows are correctly logged, overlap survives clipping at $e_{\min}$, and the pre-treatment features in $X$ capture every routing-relevant signal used by the behavior policy. The harm bound in \cref{prop:harm} is a calibration-dependent diagnostic rather than a distribution-free safety guarantee, and the bound loosens linearly with the number of collaborative protocols.

The protocol catalog is finite and manually specified, so CARL does not cover dynamic tool use, adaptive retrieval, memory, or protocol composition. Code evaluation uses SWE-bench Lite with the official harness, but Lite is a subset of SWE-bench and the auxiliary judge-based score in \cref{app:code_textual} is a sanity check rather than a substitute for execution. The main experiments use a single open-weight model family, and we do not claim contamination-free evaluation because the model's training cutoff is not auditable. Cross-model results are reported only as exploratory robustness checks.

\section*{Ethical considerations}

CARL is a routing layer over existing LLM protocols and does not generate user-facing content beyond what its constituent protocols already produce. Its direct effect is to skip collaborative calls when their estimated advantage over solo is non-positive, which reduces token usage, latency, and energy consumption relative to always-collaborate baselines. We expect the dominant risk to come from miscalibration rather than from new generative capability: a lower bound that is too tight will under-collaborate on inputs where deliberation would have caught an error, and a lower bound that is too loose will spend compute on harmful collaboration. We mitigate this through conservative calibration, audit logs of selected actions and realized advantages, and recalibration whenever the protocol catalog or base model changes.

The benchmarks used here (MATH, SWE-bench Lite, MuSiQue, 2WikiMultiHopQA) are public research datasets used under their original licenses, and we do not redistribute them. The open-weight model we use was released by its original authors; we report serving and decoding settings in the run manifest so others can replicate or audit the runs. CARL should not be deployed in high-stakes settings (medical, legal, safety-critical) without task-specific validation and human review, since the abstention decision concerns whether to invoke a collaborative protocol, not whether the final answer is correct.




% Entries for the entire Anthology, followed by custom entries
\bibliography{custom}

\clearpage


\appendix
\section*{Appendix}

\section{Benchmark and Dataset Details}
\label{app:dataset_details}

\Cref{tab:dataset_splits} reports the five splits per task used throughout the paper. Math uses MATH training data for router training and validation, and reserves MATH-500 only for final testing. QA samples disjoint splits from MuSiQue-Ans and 2WikiMultiHopQA, stratified by hop count and question type. Code uses SWE-bench Lite with the official execution-based harness. The 23 official Lite dev instances are too few to train a router, so we sample 500 training and 100 validation instances from the SWE-bench train set, with no overlap with Lite test; the 300 calibration instances are drawn from the same source under the same disjointness constraint. The final test split is the full 300 Lite test instances, and full-matrix diagnostics (CHR, catalog oracle, OPE, coverage) are computed on a stratified 100-instance audit subset drawn from Lite test. Public benchmark data are used under their original releases; we do not redistribute them and we do not claim contamination-free evaluation. We also compute a judge-based textual patch-quality score on Code, but it is reported only as an auxiliary diagnostic; the resolved indicator from the official Lite harness is the sole reward entering $R(k)$, $Y(k)$, and the DR targets.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrr}
\toprule
Task & Train & Val & Calib. & Test & Full mat. \\
\midrule
Math & 5{,}000 & 1{,}000 & 1{,}500 & 500 & 500 \\
QA   & 8{,}000 & 1{,}000 & 1{,}500 & 1{,}000 & 1{,}000 \\
Code &   500 &     100 &    300 & 300 & 100 \\
\bottomrule
\end{tabular}
\caption{Per-task split sizes. Calibration data are paired protocol executions used to fit $a_k$; Code calibration instances are drawn from the SWE-bench train set, disjoint from Lite test. The audit subset is the only Code data executed under all five protocols.}
\label{tab:dataset_splits}
\end{table}

\paragraph{Benchmark summary.}
Math uses MATH training data plus MATH-500 with exact-match reward and no retrieval. Code uses SWE-bench Lite with the official execution-based harness; the resolved indicator is the primary reward, with BM25 file retrieval over the repository as the only retrieval signal in $X$, and a judge-based textual patch-quality score reported as an auxiliary diagnostic in \cref{app:code_textual}. QA uses MuSiQue-Ans together with 2WikiMultiHopQA with token-level F1 as the reward and dataset-provided passages (no external retrieval beyond what each benchmark ships with).

\paragraph{Harness execution.}
Each Code patch is evaluated with the official SWE-bench Lite harness using a 15-minute per-instance timeout. Patches that fail to apply, run-time errors, and timeouts are all counted as unresolved. Final policy evaluation requires one harness run per Lite test instance per method per seed. The audit subset additionally requires one harness run per protocol per instance per seed, so the audit contributes $100\times 5\times 5=2{,}500$ harness executions (about 60\% of total Code harness wall-clock cost).

\section{Pre-treatment Features and Excluded Mediators}
\label{app:features}

\Cref{tab:features} lists the routing covariates that enter $X$ and the post-treatment signals that are deliberately excluded. The entropy of the routing-only solo probe is computed from a single 32-token forward pass; its generated text is discarded and never reaches the protocol catalog.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lll}
\toprule
Group & Examples & In $X$? \\
\midrule
Input descriptors & length, task type, benchmark id & yes \\
Retrieval (Code) & BM25 score, top-$k$ doc length & yes \\
Difficulty signals & problem-class, task-source ids & yes \\
Solo probe proxy & entropy of 32-token routing probe & yes \\
\midrule
Solver messages & generated answers, scratchpads & no \\
Verifier critiques & verifier outputs, votes & no \\
Debate transcripts & turn-level rationales & no \\
Judge rationales & adjudication reasoning & no \\
\bottomrule
\end{tabular}
\caption{Pre-treatment features in $X$ and excluded post-treatment mediators.}
\label{tab:features}
\end{table}

\section{Protocol Details}
\label{app:protocol_details}

The finite protocol catalog is fixed before data collection and matches the protocol IDs used in the main text. Each treatment definition specifies the model identifier, prompts, roles, decoding parameters, communication topology, turn budget, stopping rule, and aggregation rule. Generated solver messages, verifier critiques, debate transcripts, and judge rationales are post-treatment variables and not used as routing covariates. Prompts, decoding parameters, and stopping rules are in the code release. Each treatment definition is SHA-256 hashed over all prompts, decoding parameters, and checkpoint identifiers before data collection, and the treatment-definition audit log is stored with each episode.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrr}
\toprule
Protocol & Task & In tok. & Out tok. & \$/ep. \\
\midrule
\texttt{SOLO}            & Math/QA & 0.65K & 0.45K & 0.0009 \\
\texttt{SOLO}            & Code    & 9.0K  & 0.6K  & 0.0027 \\
\texttt{SELF\_REFLECT}   & Math/QA & 1.3K  & 0.9K  & 0.0017 \\
\texttt{SELF\_REFLECT}   & Code    & 10.4K & 1.2K  & 0.0040 \\
\texttt{VERIFY}          & Math/QA & 2.1K  & 1.4K  & 0.0027 \\
\texttt{VERIFY}          & Code    & 19.8K & 1.8K  & 0.0066 \\
\texttt{DEBATE\_2}       & Math/QA & 4.8K  & 3.0K  & 0.0059 \\
\texttt{DEBATE\_2}       & Code    & 29.1K & 4.2K  & 0.0122 \\
\texttt{PROPOSE\_VERIFY} & Math/QA & 5.6K  & 3.5K  & 0.0069 \\
\texttt{PROPOSE\_VERIFY} & Code    & 34.5K & 4.8K  & 0.0143 \\
\midrule
\multicolumn{4}{l}{Calibration matrix (paired runs)} & $\approx$\$70 \\
\multicolumn{4}{l}{Code audit matrix (100$\times$5$\times$5)} & $\approx$\$20 \\
\multicolumn{4}{l}{End-to-end study (incl.\ probes, judges)} & $\approx$\$560 \\
\bottomrule
\end{tabular}
\caption{Approximate token usage and replay cost. Dollar costs use illustrative accounting rates of \$0.195/M input tokens and \$1.56/M output tokens, and reflect only LLM token usage; SWE-bench Lite harness execution adds GPU/CPU compute that is not priced into these figures. Calibration matrix cost reflects Math/QA 1{,}500 + Code 300 paired protocol executions.}
\label{tab:compute_cost}
\end{table}

\section{Training and Hyperparameter Details}
\label{app:training}

\paragraph{Encoder and heads.}
The shared encoder $h_\theta$ is a frozen Qwen3.5-27B last-layer mean-pooled representation projected through a two-layer MLP (hidden size 512). Each treatment receives its own reward head, token-cost head, latency head, and advantage head, all two-layer MLPs of hidden size 256 with GELU activations and dropout 0.1.

\paragraph{Losses.}
Reward heads use binary cross-entropy on $R\in[0,1]$ for Math and Code, and MSE on F1 for QA. Token-cost and latency heads use MSE on normalized targets. The advantage head uses MSE against the shrunken DR targets $\tilde\tau_{i,k}$.

\paragraph{Folds and bootstrap.}
All nuisances are cross-fitted across five disjoint folds. The advantage-head ensemble used to estimate $\hat\sigma^{(\tau)}_k$ is five members trained on independent bootstrap resamples of the within-fold $\tilde\tau_{i,k}$. The DR variance $\hat v_{i,k}$ in $\rho_{i,k}$ is estimated from the same five folds.

\paragraph{Optimization and hyperparameters.}
We use AdamW with learning rate $2\times 10^{-4}$, weight decay $10^{-2}$, and batch size 256. Heads train for at most 40 epochs with early stopping on validation cost-aware utility. The probe ratio is $\epsilon_{\mathrm{probe}}=0.15$, the behavior-policy floor is $b_{\min}=0.02$, the propensity clip is $e_{\min}=0.02$, the calibration level is $\delta=0.1$, the LCB risk is $\kappa=1.0$, the kNN neighborhood is $K_{\mathrm{NN}}=50$, and the shrinkage scale is $\gamma=0.5$. All values are fixed on the validation calibration split before the final test, and are not retuned on the audit matrix or the final test split.

\paragraph{Stochastic decoding.}
The base model uses temperature 0.0 for solo answers and verifier critiques to keep $R(0)$ reproducible on the calibration matrix. Debate proposers use temperature 0.7 with a fixed per-seed PRNG key. When deterministic decoding is unavailable for a protocol, paired calibration entries are averaged over three executions to reduce sampling noise.

\section{Baseline Details}
\label{app:baseline_details}

\Cref{tab:baselines_appendix} summarizes baselines and notes whether each can select the solo abstention action.

\begin{table*}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llll}
\toprule
Baseline & Selection signal & Can select $T_0$? & Key distinction from CARL \\
\midrule
Always-SOLO / VERIFY / DEBATE & None & Fixed by baseline & No input-adaptive selection \\
Best-Fixed & Validation-best single protocol & Fixed by selected protocol & No input-adaptive selection \\
Solo-SelfConsist./ Solo-Rerank & Majority / reranking & Solo-only & No collaborative protocol invoked \\
MasRouter-style & Cascaded controller, no DR & No & No solo option; no causal target \\
MasRouter-style+Solo & Cascaded controller, no DR & Yes & Same action space; no causal correction \\
CascadeDebate-style & Solo confidence threshold & Yes & Confidence trigger, not effect \\
Confidence-Trigger & Solo confidence / disagreement & Yes & Single collaborative protocol \\
Adaptive-Orch.+Abs.\ & Logged utility classifier & Yes & No DR causal correction \\
DM-Greedy & Direct model $\hat m_k(x)$ & Yes & No causal correction; no pessimism \\
Causal-Routing-style & DR regret minimization & No & Absolute utility target \\
Causal-Routing-style+Solo & DR regret minimization & Yes & Absolute utility, not advantage \\
DR-Greedy & DR $\tilde\tau_k$, greedy & Yes & No calibrated pessimism \\
DR-LCB (absolute) & DR + LCB on abs.\ utility & Yes & LCB on absolute utility \\
Outcome-Reg.+LCB & Direct $\hat m_k$ + LCB & Yes & No DR; LCB on direct contrast \\
Naive-Obs.\ & Logged mean & Yes & No causal correction \\
\bottomrule
\end{tabular}
\caption{Baseline summary. The +Solo variants of MasRouter-style and Causal-Routing-style give those methods the same abstention action space as CARL.}
\label{tab:baselines_appendix}
\end{table*}

\paragraph{MasRouter-style and +Solo.}
A cascaded controller with a stage-1 binary head (collaborate vs.\ not) and a stage-2 protocol selector. Both stages train on logged cost-aware utility with the same features and capacity as CARL. The +Solo variant exposes $T_0$ to the selector at inference, matching CARL's action space.

\paragraph{CascadeDebate-style.}
Solo runs first; if its confidence falls below a validation-tuned threshold, VERIFY runs; if VERIFY confidence remains below a second threshold, DEBATE\_2 runs. Thresholds are tuned on validation utility.

\paragraph{Causal-Routing-style and +Solo.}
A causal-routing objective on protocol-level utility. The +Solo variant treats $T_0$ as a selectable action with target $\hat m_0(x)$. Unlike CARL, the objective is absolute utility rather than the one-vs-solo advantage $\hat m_k(x)-\hat m_0(x)$, and there is no calibrated lower bound on the advantage.

\section{Full Main Table with Additional Baselines}
\label{app:full_main_table}

\Cref{tab:main_full} extends \cref{tab:main} with the four baselines moved here in the main paper (Confidence-Trigger, Causal-Routing-style, Causal-Routing-style+Solo, Naive-Obs.). None of them is competitive with CARL, but reporting them keeps the comparison auditable.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Math} & \multicolumn{2}{c}{Code} & \multicolumn{2}{c}{QA} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Method & Util & CHR & Util & CHR & Util & CHR \\
\midrule
Naive-Obs.            & .52 & .26 & .22 & .31 & .48 & .29 \\
Confidence-Trigger    & .55 & .18 & .28 & .23 & .50 & .21 \\
Causal-Routing-style  & .56 & .16 & .29 & .20 & .51 & .18 \\
... +Solo             & .57 & .14 & .30 & .18 & .52 & .16 \\
\bottomrule
\end{tabular}
\caption{Additional baselines on the same three task families. Code CHR uses the 100-instance SWE-Lite audit subset.}
\label{tab:main_full}
\end{table}

\section{Code Textual Patch-Quality Diagnostic}
\label{app:code_textual}

The auxiliary judge-based patch-quality score is computed by prompting the same model family to rate each generated patch on a 5-point rubric covering localization, edit minimality, and apparent correctness, normalized to $[0,1]$. \Cref{tab:code_textual} compares textual score and resolved indicator for the main methods on Lite test. The textual score is consistently higher than the resolved indicator and ranks methods similarly, but it does not require execution. We treat it as a sanity check rather than a substitute for the official harness.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lcc}
\toprule
Method & Tex. & Res. \\
\midrule
Always-SOLO            & .42 & .28 \\
Best-Fixed             & .47 & .30 \\
MasRouter-style+Solo   & .52 & .36 \\
CascadeDebate-style    & .51 & .35 \\
DR-LCB (absolute)      & .53 & .37 \\
CARL                   & \textbf{.58} & \textbf{.39} \\
\bottomrule
\end{tabular}
\caption{SWE-bench Lite test: judge textual patch-quality score (Tex.) vs.\ official resolved rate (Res.).}
\label{tab:code_textual}
\end{table}

\section{Calibration and Selected-Action Coverage}
\label{app:calibration}

\Cref{tab:selected_action_cal} reports per-treatment selected-action lower-bound coverage, conditional on each protocol being selected. SOLO has coverage 1 by construction. Coverage on Code is slightly below the nominal 90\% target for debate-style treatments, consistent with wider DR variance on the audit subset.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
Task & SR & VER & DEB & PV & SOLO \\
\midrule
Math & .93 & .91 & .90 & .89 & 1.00 \\
Code & .89 & .88 & .85 & .87 & 1.00 \\
QA   & .89 & .90 & .87 & .88 & 1.00 \\
\bottomrule
\end{tabular}
\caption{Per-treatment lower-bound coverage of CARL conditional on the protocol being selected. Math and QA use the final test matrix; Code uses the 100-instance audit subset.}
\label{tab:selected_action_cal}
\end{table}

\section{Confounding, OPE, and Sensitivity Diagnostics}
\label{app:sensitivity}
\label{app:bootstrap}

\paragraph{Propensity overlap.}
\Cref{tab:overlap} reports the mean assignment probability, the 5th percentile (a worst-case overlap proxy), the effective sample size after clipping, and the largest realized propensity-weight clip ratio under $e_{\min}=0.02$.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccc}
\toprule
Treatment & Mean prop. & 5th pct.\ & ESS/$n$ & Max clip \\
\midrule
\texttt{SOLO}            & 0.38 & 0.18 & 0.82 & 5.0  \\
\texttt{SELF\_REFLECT}   & 0.17 & 0.07 & 0.74 & 8.0  \\
\texttt{VERIFY}          & 0.21 & 0.08 & 0.76 & 8.0  \\
\texttt{DEBATE\_2}       & 0.12 & 0.04 & 0.61 & 10.0 \\
\texttt{PROPOSE\_VERIFY} & 0.12 & 0.04 & 0.59 & 10.0 \\
\bottomrule
\end{tabular}
\caption{Propensity overlap on training logs.}
\label{tab:overlap}
\end{table}

\paragraph{Placebo treatments.}
\Cref{tab:placebo} reports placebo treatments that should not look helpful if the advantage learner is well calibrated. ``Random critique'' inserts noise; ``Shuffled verifier'' swaps verifier inputs across items; ``Extra summary agent'' adds a redundant turn.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lccc}
\toprule
Placebo & Est.\ adv.\ & Audit gain & Sel.\ \% \\
\midrule
Random critique     & $-0.03$ & $-0.04$ & 2\% \\
Shuffled verifier   & $-0.05$ & $-0.06$ & 1\% \\
Extra summary agent & $-0.01$ & $-0.02$ & 4\% \\
\bottomrule
\end{tabular}
\caption{Placebo treatments: estimated advantage, true audit gain, and CARL selection rate.}
\label{tab:placebo}
\end{table}

\paragraph{Bootstrap standard errors.}
\Cref{tab:confounding_se} reports bootstrap SE ($B=1000$) for the logged and audit contrasts in \cref{tab:confounding}, restructured so SEs sit in their own columns.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}
\toprule
Task & Treatment & Log est. & Log SE & Audit est. & Audit SE \\
\midrule
\multirow{4}{*}{Math}
& SR  & $-0.012$ & .010 & $+0.024$ & .009 \\
& VER & $-0.071$ & .018 & $+0.038$ & .017 \\
& DEB & $-0.029$ & .019 & $+0.018$ & .018 \\
& PV  & $-0.046$ & .017 & $+0.027$ & .015 \\
\midrule
\multirow{4}{*}{Code}
& SR  & $+0.003$ & .012 & $+0.011$ & .011 \\
& VER & $-0.054$ & .022 & $+0.022$ & .020 \\
& DEB & $-0.041$ & .021 & $+0.008$ & .014 \\
& PV  & $-0.061$ & .020 & $+0.014$ & .013 \\
\midrule
\multirow{4}{*}{QA}
& SR  & $+0.019$ & .010 & $+0.013$ & .009 \\
& VER & $+0.008$ & .011 & $+0.021$ & .010 \\
& DEB & $-0.058$ & .018 & $-0.036$ & .017 \\
& PV  & $-0.030$ & .013 & $-0.014$ & .012 \\
\bottomrule
\end{tabular}
\caption{Bootstrap SE ($B{=}1000$) for the contrasts in \cref{tab:confounding}. Sign reversals for VERIFY on Math and Code exceed two SE in magnitude.}
\label{tab:confounding_se}
\end{table}

\paragraph{OPE bootstrap SE.}
\Cref{tab:ope_se} reports SEs for the off-policy evaluation errors in \cref{tab:ope}.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{10pt}
\begin{tabular}{lcccc}
\toprule
Task & DM & IPW & SNIPW & DR \\
\midrule
Math & .005 & .006 & .005 & .004 \\
Code & .009 & .010 & .008 & .007 \\
QA   & .006 & .007 & .005 & .005 \\
\bottomrule
\end{tabular}
\caption{Bootstrap SE ($B{=}1000$) for the OPE errors in \cref{tab:ope}.}
\label{tab:ope_se}
\end{table}

\paragraph{Rosenbaum-style hidden-confounding sensitivity.}
\Cref{tab:sensitivity_analysis} reports a Rosenbaum-style sensitivity analysis for the VERIFY advantage on Math. Larger $\Gamma$ corresponds to stronger unobserved assignment bias. The lower bound crosses zero only beyond $\Gamma\approx 1.5$.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{8pt}
\begin{tabular}{lccc}
\toprule
$\Gamma$ & Lower & Point & Upper \\
\midrule
1.00 (none) & $+0.02$ & $+0.04$ & $+0.06$ \\
1.25 & $+0.00$ & $+0.04$ & $+0.08$ \\
1.50 & $-0.01$ & $+0.04$ & $+0.09$ \\
2.00 & $-0.04$ & $+0.04$ & $+0.12$ \\
\bottomrule
\end{tabular}
\caption{Rosenbaum-style sensitivity for VERIFY advantage on Math.}
\label{tab:sensitivity_analysis}
\end{table}

\paragraph{Behavior policy logging.}
During training, 15\% of episodes come from the randomized probe $q_\psi$ and 85\% from exploitation $\pi_\phi$, with the behavior-policy floor $b_{\min}=0.02$ enforced on every $(x,k)$. The exploitation policy assigns harder inputs (estimated from pre-treatment solo-probe entropy and difficulty labels) disproportionately to verification and debate, which produces the sign-reversed logged contrasts in \cref{tab:confounding}. The DR interpretation depends on the probe propensities being logged correctly and on the routing-relevant pre-treatment features in \cref{tab:features} being sufficient.

\paragraph{$b_{\min}$ sensitivity.}
\Cref{tab:bmin} sweeps the behavior-policy floor $b_{\min}$. Lowering $b_{\min}$ to zero would in principle break overlap on expensive protocols on hard inputs, but the CARL policy is fairly stable across $b_{\min}\in\{0.01,0.02,0.05\}$ because $\hat\sigma^{(\tau)}_k$ widens automatically when local support is thin.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{8pt}
\begin{tabular}{lccc}
\toprule
$b_{\min}$ & Util & CHR & Cov. \\
\midrule
0.01 & .57 & .12 & .87 \\
0.02 (default) & \textbf{.58} & \textbf{.10} & \textbf{.90} \\
0.05 & .58 & .10 & .91 \\
\bottomrule
\end{tabular}
\caption{Sensitivity to the behavior-policy floor $b_{\min}$, averaged over Math and QA.}
\label{tab:bmin}
\end{table}

\section{Cost Coefficient Sweep}
\label{app:lambda_sweep}

\Cref{tab:lambda_sweep} reports CARL utility and CHR over a grid of $(\lambda,\mu)$, averaged over Math and QA. The shared encoder, reward head, token-cost head, and latency head are not retrained for each grid cell; each cell only recombines $\hat m_k$, rebuilds the DR scores, refits the advantage head, and recalibrates $a_k$. CARL stays near the validation-tuned operating point $(\lambda,\mu)=(0.05,0.02)$ across the grid, and the abstention rate increases monotonically with $\lambda$.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{cccccc}
\toprule
$\lambda$ & $\mu$ & Util & CHR & Abst.\ & Tok./ep \\
\midrule
0.01 & 0.01 & .57 & .12 & .35 & 5.4K \\
0.03 & 0.02 & .58 & .11 & .42 & 4.7K \\
0.05 & 0.02 & \textbf{.58} & \textbf{.10} & .48 & 4.1K \\
0.05 & 0.05 & .57 & .09 & .55 & 3.7K \\
0.10 & 0.05 & .56 & .07 & .63 & 3.1K \\
\bottomrule
\end{tabular}
\caption{Cost coefficient sweep, averaged over Math and QA. The shared encoder and nuisance heads are trained once; each grid cell refits the advantage head and recalibrates the LCB policy from cached nuisance predictions.}
\label{tab:lambda_sweep}
\end{table}

\section{Additional Experimental Results}
\label{app:extra_results}

\subsection{Per-seed Utility}
\label{app:per_seed}

\Cref{tab:per_seed} reports per-seed utility for CARL against MasRouter-style+Solo and Outcome-Reg.+LCB across the three task families. Paired $p$-values are diagnostics only because there are five training seeds.

\begin{table*}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccccc}
\toprule
Task & Method & S1 & S2 & S3 & S4 & S5 & Mean & $p$ vs.\ CARL \\
\midrule
\multirow{3}{*}{Math}
 & CARL                & .62 & .60 & .61 & .62 & .60 & .610 & -- \\
 & MasRouter+Solo      & .58 & .56 & .57 & .58 & .56 & .570 & .003 \\
 & Outcome-Reg.+LCB    & .58 & .56 & .57 & .57 & .56 & .568 & .004 \\
\midrule
\multirow{3}{*}{Code}
 & CARL                & .36 & .34 & .35 & .35 & .35 & .350 & -- \\
 & MasRouter+Solo      & .33 & .31 & .32 & .32 & .32 & .320 & .008 \\
 & Outcome-Reg.+LCB    & .33 & .31 & .32 & .32 & .32 & .320 & .008 \\
\midrule
\multirow{3}{*}{QA}
 & CARL                & .56 & .54 & .55 & .55 & .55 & .550 & -- \\
 & MasRouter+Solo      & .54 & .52 & .53 & .53 & .53 & .530 & .005 \\
 & Outcome-Reg.+LCB    & .53 & .51 & .52 & .52 & .52 & .520 & .002 \\
\bottomrule
\end{tabular}
\caption{Per-seed cost-aware utility. Paired $p$-values compare each baseline against CARL across five seeds.}
\label{tab:per_seed}
\end{table*}

\subsection{Risk Parameter Sweep}

\Cref{tab:kappa} sweeps the LCB risk parameter $\kappa$. Larger $\kappa$ reduces CHR and tokens but eventually lowers utility by abstaining too often. The validation-selected $\kappa=1.0$ sits near the elbow.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{8pt}
\begin{tabular}{ccccc}
\toprule
$\kappa$ & Util & CHR & Abst.\ & Tok./ep \\
\midrule
0.0 & .59 & .21 & .22 & 5.7K \\
0.5 & .59 & .14 & .35 & 4.8K \\
1.0 & \textbf{.58} & .10 & .48 & 4.1K \\
1.5 & .56 & .06 & .59 & 3.4K \\
2.0 & .53 & \textbf{.03} & .68 & 2.8K \\
\bottomrule
\end{tabular}
\caption{Effect of $\kappa$, averaged over Math and QA. $\kappa=1.0$ is the validation-tuned default.}
\label{tab:kappa}
\end{table}

\subsection{Probe-budget Sensitivity}

\Cref{tab:probe_budget} sweeps the probe ratio $\epsilon_{\mathrm{probe}}$. Returns to randomization diminish past 15\%, and 0\% probing reduces to the observational case where overlap is enforced only by clipping.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lcccc}
\toprule
$\epsilon_{\mathrm{probe}}$ & Util & CHR & Recov. & Extra tok \\
\midrule
0\%  & .53 & .15 & 27\% & 1.00$\times$ \\
5\%  & .55 & .13 & 41\% & 1.08$\times$ \\
10\% & .57 & .11 & 53\% & 1.16$\times$ \\
15\% & \textbf{.58} & \textbf{.10} & \textbf{60\%} & 1.24$\times$ \\
25\% & .58 & .09 & 60\% & 1.43$\times$ \\
\bottomrule
\end{tabular}
\caption{Probe ratio sensitivity, averaged over Math and QA. Per-task catalog-oracle recovery is in \cref{tab:oracle_gain}.}
\label{tab:probe_budget}
\end{table}

\subsection{Compute-matched Solo Comparison}

\Cref{tab:compute_matched} compares CARL with solo strategies that consume comparable token budgets but never invoke $T_1$--$T_4$. Reward-per-1K-tokens is a normalized efficiency measure.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}
\toprule
Method & Util & Reward & Tok./ep & R/1K \\
\midrule
Solo-SelfConsist.\ & .53 & .58 & 4.2K & .138 \\
Solo-Rerank        & .54 & .59 & 4.0K & .148 \\
MasRouter+Solo     & .55 & .60 & 5.2K & .115 \\
Adapt.+Abs.\        & .55 & .60 & 5.1K & .118 \\
CARL               & \textbf{.58} & \textbf{.62} & 4.1K & \textbf{.151} \\
\bottomrule
\end{tabular}
\caption{Compute-matched comparison, averaged over Math and QA.}
\label{tab:compute_matched}
\end{table}

\subsection{Cost Decomposition Ablation}

\Cref{tab:decomp} reports the cost-decomposition ablation, including held-out cross-$\lambda$ transfer where the cost coefficients at test time differ from training.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}
\toprule
Variant & Util & CHR & ECE & PTC \\
\midrule
Single utility head  & .54 & .15 & .10 & .74 \\
Reward + token only  & .56 & .12 & .07 & .79 \\
Full decomposed (CARL) & \textbf{.58} & \textbf{.10} & \textbf{.05} & \textbf{.81} \\
\midrule
\multicolumn{5}{l}{\textit{Cross-$\lambda$ transfer}} \\
Single utility head  & .50 & .19 & -- & -- \\
Full decomposed (CARL) & \textbf{.55} & \textbf{.12} & -- & -- \\
\bottomrule
\end{tabular}
\caption{Cost decomposition ablation, averaged over Math and QA. Cross-$\lambda$ transfer evaluates held-out cost coefficients $\lambda\in\{0.01,0.03,0.05,0.10\}$.}
\label{tab:decomp}
\end{table}

\subsection{Selected Treatment Distribution}

\Cref{tab:selected_treatments} reports the distribution of CARL-selected actions on the final test set.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
Task & SOLO & SR & VER & DEB & PV \\
\midrule
Math (MATH-500)    & 42\% & 12\% & 29\% & 7\% & 10\% \\
Code (SWE Lite)    & 51\% & 14\% & 22\% & 4\% & 9\%  \\
QA (MuSiQue+2Wiki) & 55\% & 13\% & 18\% & 6\% & 8\%  \\
\bottomrule
\end{tabular}
\caption{CARL selected treatment distribution. Code uses the 300-instance Lite test set.}
\label{tab:selected_treatments}
\end{table}

\subsection{Distribution Shift}

\Cref{tab:shift} reports controlled distribution-shift diagnostics. These are finite-catalog stress tests and not guarantees under arbitrary deployment shift.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{MR+Solo} & \multicolumn{2}{c}{CARL} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Setting & Util & CHR & Util & CHR \\
\midrule
In-domain (Math)    & .57 & .17 & \textbf{.61} & \textbf{.08} \\
Harder math         & .51 & .24 & \textbf{.55} & \textbf{.12} \\
Longer SWE issues   & .29 & .29 & \textbf{.33} & \textbf{.14} \\
Noisy retrieval QA  & .48 & .26 & \textbf{.51} & \textbf{.13} \\
\bottomrule
\end{tabular}
\caption{Distribution-shift evaluation. Math uses the hardest MATH quartile; Code uses longer SWE Lite issues with extended retrieved context; QA uses perturbed BM25 retrieval.}
\label{tab:shift}
\end{table}



\subsection{Protocol Drift}

\Cref{tab:drift} reports treatment-drift stress tests in which protocol definitions or model behavior change after calibration. ``No recal.'' applies the original selector to the shifted protocol; ``Recal.'' refits $a_k$ on 200 probe episodes from the shifted setting.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
Shift & NR Util & NR CHR & RC Util & RC CHR \\
\midrule
Verifier prompt wording & .55 & .14 & .57 & .11 \\
Judge prompt wording    & .55 & .13 & .57 & .11 \\
Decoding temp.\ +0.2    & .54 & .15 & .56 & .12 \\
Sibling checkpoint      & .53 & .17 & .55 & .13 \\
\bottomrule
\end{tabular}
\caption{Treatment-drift stress tests, averaged over Math and QA.}
\label{tab:drift}
\end{table}

\subsection{Sequential Deployment}

\Cref{tab:sequential} reports a sequential deployment diagnostic on Math, showing how performance changes as probe-supported calibration data accumulate over 500 deployment episodes. The probe rate is gradually reduced after the initial collection.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}
\toprule
Episodes & Util & CHR & Cov. & PTC & Probe \\
\midrule
1--100   & .54 & .19 & .82 & .73 & 25\% \\
101--200 & .56 & .14 & .85 & .79 & 20\% \\
201--300 & .58 & .12 & .88 & .81 & 17\% \\
301--400 & .59 & .10 & .90 & .83 & 15\% \\
401--500 & .61 & .08 & .91 & .86 & 15\% \\
\bottomrule
\end{tabular}
\caption{CARL over 500 sequential deployment episodes on Math.}
\label{tab:sequential}
\end{table}

\subsection{Stress Tests}
\label{app:stress}

The main text mentions three stress settings: injected protocol failures (a fraction of collaborative protocol executions return errors and fall back to the solo answer), rate-limit-style missing responses (a fraction of agent calls return empty, simulating a busy serving stack), and adversarial distractor critiques (a synthetic verifier inserts plausible but incorrect critiques into the verifier protocol). \Cref{tab:stress} reports CARL utility and CHR under each setting at a 10\% per-call rate, averaged over Math and QA. CARL's calibrated abstention absorbs most of the lost utility by widening $\hat\sigma^{(\tau)}_k$ on inputs affected by the stress, keeping CHR close to its default value at the cost of a small utility drop.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lcc}
\toprule
Setting & CARL Util & CARL CHR \\
\midrule
No stress (default)               & \textbf{.58} & \textbf{.10} \\
Injected protocol failures        & .55 & .11 \\
Rate-limit missing responses      & .56 & .10 \\
Adversarial distractor critiques  & .54 & .13 \\
\bottomrule
\end{tabular}
\caption{CARL under three stress settings, averaged over Math and QA. Each stress is applied at a 10\% per-call rate.}
\label{tab:stress}
\end{table}

\section{Cross-Model Robustness}
\label{app:cross_model}

The main experiments use \texttt{Qwen/Qwen3.5-27B} throughout, which 
raises a reasonable concern: are CARL's gains specific to this 
checkpoint, or do they extend to base models with different reasoning 
profiles? We report a controlled cross-model robustness study here. 
These results are exploratory: cross-model runs are not used for any main 
claim, the protocol catalog is held fixed across models, all 
hyperparameters (probe ratio, $\delta$, $\kappa$, $b_{\min}$, 
$\gamma$, $K_{\mathrm{NN}}$) are inherited from the Qwen-27B-tuned 
defaults rather than re-tuned per model, and each (model, task) cell 
is run with a single seed. Per-cell numbers should therefore be read 
as directional evidence rather than precise effect sizes.

\subsection{Setup}

We evaluate CARL with two additional base models drawn from different 
families and capability tiers: a smaller open-weight model 
(\texttt{Llama-3.1-8B-Instruct}) and a larger open-weight model 
(\texttt{Qwen2.5-72B-Instruct}). We use Qwen2.5-72B rather than a 
larger Qwen3.5 variant because the Qwen3.5 family above 27B uses 
sparse/mixture-of-experts architectures (e.g., 
\texttt{Qwen3.5-122B-A10B}), which complicate the latency/token cost 
accounting underlying our utility definition. For each model we rerun 
the full CARL pipeline---behavior-policy logging with the same 
$\epsilon_{\mathrm{probe}}=0.15$, cross-fitted nuisance training, 
DR advantage estimation, and conformal calibration---using 30\% of 
the original training logs per task together with the full 
calibration matrix (1{,}500/1{,}500/300 for Math/QA/Code). Logging is 
reduced because behavior-policy data collection dominates router 
training cost, while the calibration matrix is retained at full size 
to avoid conflating model-transfer effects with conformal-quantile 
noise. Each protocol's prompts, roles, decoding parameters, and 
stopping rules are unchanged; only the checkpoint behind every 
protocol is swapped. The latency reference budget in $Y$ is 
renormalized per model so that token and latency costs remain on the 
unit scale, which means absolute utility values are not directly 
comparable across models---only the within-model ranking of methods 
is meaningful. We compare against the two strongest non-CARL 
baselines from \cref{tab:main}: MasRouter-style+Solo and 
DR-LCB (absolute utility).

\subsection{Main Cross-Model Results}

\Cref{tab:crossmodel_main} reports cost-aware utility and CHR for 
each model on the three task families. The qualitative ranking 
CARL $\succeq$ DR-LCB $\succeq$ MR+Solo holds on every (model, task) 
cell, with the utility margin over DR-LCB falling in the 0.02--0.04 
range and the CHR reduction in the 0.04--0.08 range. The margins 
contract on Qwen2.5-72B, where within-model SOLO performance is 
already high under its own normalization and the headroom for any 
collaborative protocol to add positive advantage is smaller; on 
Llama-3.1-8B, within-model utilities are uniformly lower and CHR 
uniformly higher than the corresponding Qwen-27B cells (subject to 
the per-model normalization caveat above), consistent with the 
noisier reward signal a weaker base model produces. Because each 
cell uses a single seed, the per-cell margins should be read as 
directional rather than as calibrated effect sizes; the qualitative 
ranking is the load-bearing claim, not the specific decimal values.

\begin{table*}[h]
\centering
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular*}{\textwidth}{l@{\extracolsep{\fill}}lccc}
\toprule
Model & Method & Math Util/CHR & Code Util/CHR$^\dagger$ & QA Util/CHR \\
\midrule
\multirow{3}{*}{Llama-3.1-8B}
 & MR+Solo  & .47 / .22 & .21 / .31 & .43 / .26 \\
 & DR-LCB   & .49 / .18 & .23 / .25 & .45 / .22 \\
 & CARL     & \textbf{.53} / \textbf{.11} & \textbf{.26} / \textbf{.16} & \textbf{.48} / \textbf{.15} \\
\midrule
\multirow{3}{*}{Qwen3.5-27B (main)}
 & MR+Solo  & .57 / .17 & .32 / .22 & .53 / .19 \\
 & DR-LCB   & .58 / .14 & .33 / .17 & .53 / .16 \\
 & CARL     & \textbf{.61} / \textbf{.08} & \textbf{.35} / \textbf{.11} & \textbf{.55} / \textbf{.11} \\
\midrule
\multirow{3}{*}{Qwen2.5-72B}
 & MR+Solo  & .63 / .14 & .38 / .19 & .58 / .16 \\
 & DR-LCB   & .64 / .12 & .39 / .15 & .59 / .13 \\
 & CARL     & \textbf{.66} / \textbf{.07} & \textbf{.41} / \textbf{.10} & \textbf{.61} / \textbf{.09} \\
\bottomrule
\multicolumn{5}{l}{\footnotesize $^\dagger$Code Util is on the full 300-instance Lite test split; Code CHR is on the 100-instance audit subset.}\\
\end{tabular*}
\caption{Cross-model robustness. Each cell shows cost-aware utility 
and harmful-collaboration rate. Absolute utility values are not 
directly comparable across models because the latency reference 
budget is renormalized per model (see Setup); the within-model 
method ranking is the intended comparison. Hyperparameters are not 
re-tuned per model, and each cell uses a single seed.}
\label{tab:crossmodel_main}
\end{table*}

\subsection{Calibration Transfer}

A more pointed concern is whether CARL's calibrated lower bound 
remains well-calibrated when the base model changes. We address this 
by reporting selected-action coverage and treatment-effect ECE for 
each model in \cref{tab:crossmodel_coverage}. Coverage falls within 
2--6 percentage points of the nominal 90\% target, with the largest 
undercoverage on \texttt{Llama-3.1-8B} for Code (0.84), where the 
reduced training-log budget and the higher reward noise of the 
weaker base model together produce the widest residual distribution. 
On Math and QA, Llama-3.1-8B undercoverage is mild (2--4 points), 
and on Qwen2.5-72B coverage is essentially at the target. ECE 
remains below 0.10 except for Llama-3.1-8B on Code (0.11), 
consistent with the same noise pattern. These results suggest that 
the LCB construction adapts to noisier residuals by widening $a_k$ 
rather than silently violating coverage, but the Code/Llama-8B cell 
indicates the adaptation is imperfect under reduced logs and serves 
as a useful failure mode to flag.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Math} & \multicolumn{2}{c}{Code} & \multicolumn{2}{c}{QA} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Model & Cov & ECE & Cov & ECE & Cov & ECE \\
\midrule
Llama-3.1-8B   & .88 & .07 & .84 & .11 & .86 & .08 \\
Qwen3.5-27B    & .91 & .05 & .87 & .08 & .89 & .06 \\
Qwen2.5-72B    & .91 & .04 & .88 & .07 & .90 & .05 \\
\bottomrule
\end{tabular}
\caption{Cross-model calibration. Coverage is selected-action 
lower-bound coverage; ECE is treatment-effect calibration error. 
Target coverage is 90\%. The Llama-3.1-8B Code cell undercovers by 
6 points, the only cell outside the 2--4 point band.}
\label{tab:crossmodel_coverage}
\end{table}

\subsection{Selected-Action Distribution Shifts}

The CARL-selected action mix shifts with model capability in a way 
that is qualitatively consistent with the collaboration-advantage 
estimand. On Llama-3.1-8B, SOLO is selected on 31\% of Math instances 
(compared with 42\% for Qwen3.5-27B and 55\% for Qwen2.5-72B), and 
verification-style protocols take more mass, reflecting the larger 
upside of an independent checker when the base solver is weaker. On 
Qwen2.5-72B, SOLO dominates further on Math and QA (55\% and 63\%), 
and CHR drops correspondingly, since the headroom for any 
collaborative protocol to add positive advantage is smaller. 
\Cref{tab:crossmodel_selection} reports the full breakdown. The 
direction is what the estimand predicts: as $m_0^\star(x)$ increases, 
$\tau_k(x)$ shrinks for most $k$ and the LCB policy abstains more 
often. We do not claim this monotone shift is a calibrated 
quantitative prediction, only that the qualitative direction is 
consistent.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccccc}
\toprule
Model & Task & SOLO & SR & VER & DEB & PV \\
\midrule
\multirow{3}{*}{Llama-3.1-8B}
 & Math & 31\% & 15\% & 36\% & 8\%  & 10\% \\
 & Code & 39\% & 16\% & 27\% & 6\%  & 12\% \\
 & QA   & 42\% & 16\% & 24\% & 8\%  & 10\% \\
\midrule
\multirow{3}{*}{Qwen2.5-72B}
 & Math & 55\% & 10\% & 22\% & 5\%  & 8\%  \\
 & Code & 60\% & 12\% & 18\% & 3\%  & 7\%  \\
 & QA   & 63\% & 11\% & 16\% & 4\%  & 6\%  \\
\bottomrule
\end{tabular}
\caption{CARL-selected action distribution by model. The Qwen3.5-27B 
distribution is in \cref{tab:selected_treatments}. SOLO selection 
rises with base-model capability, consistent with shrinking 
collaboration advantage as the solo solver improves.}
\label{tab:crossmodel_selection}
\end{table}

\subsection{Discussion and Limitations}

Three observations are worth recording. First, the ranking 
CARL $\succeq$ DR-LCB $\succeq$ MR+Solo is preserved across all 
three models and three task families in the cells we ran, providing 
preliminary evidence that CARL's gains are not unique to the main 
Qwen3.5-27B checkpoint. Second, CARL's calibrated coverage and ECE 
stay within an acceptable band even when the base model is markedly 
weaker, except on the Llama-3.1-8B Code cell where undercoverage 
reaches 6 points and ECE reaches 0.11; we flag this rather than 
claim universal calibration robustness. Third, the action mix shifts 
in the direction the collaboration-advantage estimand predicts: 
stronger models abstain more, weaker models collaborate more.

Several limitations apply specifically to this section. 
\textbf{(i)} We hold the protocol catalog fixed across models. In 
principle a 72B verifier paired with an 8B solver might shift the 
advantage structure substantially, and we do not study 
heterogeneous-model catalogs here. 
\textbf{(ii)} Hyperparameters are inherited from the Qwen3.5-27B 
defaults; some per-model tuning would likely improve absolute 
numbers, especially for Llama-3.1-8B. 
\textbf{(iii)} Cross-model runs use 30\% of the training logs and a 
single seed per (model, task) cell, so the per-cell numbers are not 
accompanied by bootstrap standard errors and should be read as 
directional evidence rather than as precise effect sizes. We do not 
report significance tests for this appendix-only single-seed study; 
sign-of-effect comparisons against the main Qwen3.5-27B rows are the 
intended use of \cref{tab:crossmodel_main}. 
\textbf{(iv)} The latency reference budget is renormalized per model, 
so absolute utility values are not comparable across the three model 
rows in \cref{tab:crossmodel_main}; only the within-model method 
ordering is. 
\textbf{(v)} The numbers in \cref{tab:crossmodel_main,tab:crossmodel_coverage,tab:crossmodel_selection} 
come from cross-model runs with the reduced-log budget described 
above; they are not a substitute for re-running the main experiments 
under each base model with full training data and multiple seeds. 
For this reason, no numerical robustness claim from cross-model 
results enters the main text.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcc}
\toprule
Component & Reused & Re-tuned per model \\
\midrule
Probe ratio $\epsilon_{\mathrm{probe}}=0.15$ & \cmark & \xmark \\
Behavior floor $b_{\min}=0.02$               & \cmark & \xmark \\
Calibration $\delta=0.1$, $\kappa=1.0$       & \cmark & \xmark \\
Shrinkage $\gamma=0.5$, $K_{\mathrm{NN}}=50$ & \cmark & \xmark \\
Protocol prompts and stopping rules          & \cmark & \xmark \\
Encoder MLP capacity, dropout                & \cmark & \xmark \\
Latency reference budget                     & \xmark & \cmark \\
\bottomrule
\end{tabular}
\caption{Hyperparameter handling in the cross-model study. All 
router-side settings are inherited from the Qwen3.5-27B-tuned 
defaults; only the latency reference budget is renormalized per 
model so that token and latency costs remain on the unit scale.}
\label{tab:crossmodel_hparams}
\end{table}

\section{Catalog-Oracle Gain Convention}
\label{app:oracle_gain}

The main text reports catalog-oracle recovery per task. The pooled cross-task average is
\[
\frac{U_{\mathrm{CARL,avg}}-U_{\mathrm{Solo,avg}}}{U_{\mathrm{Cat.Oracle,avg}}-U_{\mathrm{Solo,avg}}}
=\frac{0.50-0.41}{0.57-0.41}\approx 56\%.
\]
This is descriptive: it summarizes how much of the catalog-oracle improvement over solo is recovered under the validation-tuned cost coefficients. It is not online observable. The per-task values are in \cref{tab:oracle_gain}.

\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}
\toprule
Task & Solo & CARL & Cat.\ Oracle & Recov. \\
\midrule
Math (MATH-500)    & .51 & .61 & .68 & 59\% \\
Code (SWE Lite)    & .25 & .35 & .44 & 53\% \\
QA (MuSiQue+2Wiki) & .47 & .55 & .60 & 62\% \\
\bottomrule
\end{tabular}
\caption{Per-task catalog-oracle recovery.}
\label{tab:oracle_gain}
\end{table}

\section{Additional Theoretical Details}
\label{app:theory}

\begin{proposition}[Policy advantage-error bound]
Under finite $\Tset$, bounded outcomes, i.i.d.\ inputs, and $\sup_{x,k}|\tilde\tau_k(x)-\tau_k(x)|\le\epsilon$ for all $k\ge 1$, the greedy one-vs-solo policy $\hat\pi$ satisfies $J(\pi^\star)-J(\hat\pi)\le 2\epsilon$.
\end{proposition}

\begin{proof}
For any input $x$, let $a^\star(x)$ be the oracle action and $\hat a(x)$ the greedy action under the estimated advantages. Since $T_0$ has advantage zero by definition and every collaborative advantage is estimated within $\epsilon$, the estimated advantage of $\hat a(x)$ is at least that of $a^\star(x)$. The true advantage of $\hat a(x)$ is therefore at most $2\epsilon$ below that of $a^\star(x)$. Taking expectations over $X$ gives the result.
\end{proof}

\begin{corollary}[Compute spent on harmful collaboration]
Under the calibration condition in \cref{prop:harm}, the probability of selecting a harmful collaborative protocol is at most $\sum_{k=1}^K\delta_k$. If the maximum excess compute cost of any collaborative protocol over solo is $C_{\max}$, the expected excess compute spent on harmful collaboration is at most $C_{\max}\sum_{k=1}^K\delta_k$.
\end{corollary}

This corollary inherits the limitations of \cref{prop:harm}: it is a calibration-dependent diagnostic, not a distribution-free safety guarantee.

\section{Reproducibility and Released Assets}
\label{app:reproducibility}

The anonymized release includes protocol manifests, prompt templates, treatment hashes, split files, routing and calibration code, baseline implementations, harness invocation scripts for SWE-bench Lite, and table-generation scripts. Public benchmark data are not redistributed when the original license requires users to obtain the data from the source. The release documents the commands and configuration files needed to reproduce \cref{tab:main} from logged protocol executions and harness outputs. Cross-model runs, if included, are marked exploratory and require exact model identifiers, request dates, prompts, decoding parameters, and usage logs.

\section{Compute Resources}
\label{app:compute_resources}

The experiments do not train or fine-tune the base language model. Compute is dominated by inference-time protocol executions and, for Code, by SWE-bench Lite harness runs. Main experiments use local vLLM serving with a fixed GPU configuration, and token counts are logged for every protocol execution. \Cref{tab:compute_cost} reports per-protocol token usage, the calibration matrix cost, the Code audit matrix cost, and the larger end-to-end study budget. Cost-aware utility is computed from logged token counts using the paper's cost coefficient $\lambda$ rather than from provider billing.

\section{Broader Impacts}
\label{app:broader_impacts}

CARL may reduce unnecessary LLM calls by abstaining from collaboration when the estimated advantage is not positive, which can lower cost, latency, and energy use. It may also improve reliability by making harmful collaboration measurable and by exposing calibration diagnostics. Potential negative impacts include overreliance on calibrated lower bounds, deployment in high-stakes settings without task-specific review, and degraded behavior under distribution shift or treatment drift. Mitigations include conservative abstention thresholds, audit logs, calibration monitoring, task-specific validation, human review in high-stakes settings, and recalibration when protocol definitions or model versions change.

\section{Asset Licenses and Terms}
\label{app:asset_licenses}

The paper uses public benchmarks, open-weight model checkpoints, and standard software libraries. MATH, MATH-500, SWE-bench Lite, MuSiQue, and 2WikiMultiHopQA are cited in the main paper and used according to their public release terms. The model checkpoint and software dependencies are identified in the run manifest. The released artifacts contain CARL code, prompts, manifests, split identifiers, and evaluation scripts, but do not redistribute third-party benchmark data when redistribution is not permitted. Users of the released code are instructed to obtain benchmark assets from their original sources and comply with the corresponding licenses and terms of use.

\end{document}