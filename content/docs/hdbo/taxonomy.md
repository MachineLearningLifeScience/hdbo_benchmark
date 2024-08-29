---
layout: docs
type: minimal
modules: ["katex"]
---

#### A taxonomy of high-dimensional Bayesian Optimization {#sec:taxonomy}

We describe the field of high dimensional BO and the large number of
related publications through a refined taxonomy building on previous
work, discussing *variable selection*, *additive models*, *trust
regions*, *linear embeddings*, *non-linear embeddings*, *gradient
information*, *structured spaces*, and others in turn. While
encompassing taxonomies over fields may initially appear ill-advised
[@Wilkins:PhilLang:1668 pp.22], we highlight commonalities in strategies
that give structure to the HBDO problem-space.

We expand previous surveys
[@BinoisWycoff:HDGPs:2022; @SantoniDoerr:HDBO:2023] and identify a finer
taxonomy, of seven method groups and new families of *structured spaces*
(i.e.methods that work directly on mixed representations, or Riemannian
manifolds, previously categorized as *non-linear embeddings*), and
methods that rely on predicted *gradient information*. This new
separation emphasizes the heterogeneous nature of discrete solvers: some
optimizers work directly on discrete space (*structured spaces*), while
others optimize using latent representations (*non-linear embeddings*);
gradient-based methods are separated to show alternatives when
first-order information is available.
Fig. [1](#fig:hdbo_timeline){reference-type="ref"
reference="fig:hdbo_timeline"} presents a timeline of HDBO methods,
split into these families, and all methods are detailed in supplementary
Table [1](#tab:appendix:full_taxonomy){reference-type="ref"
reference="tab:appendix:full_taxonomy"}; methods are grouped according
to their most dominant feature.

#### Variable selection

To solve a high-dimensional problem, one approach is to focus on a
subset of variables of high interest.[^3] One selects the variables
either by using domain expertise, or by *Automatic Relevance Detection*
(ARD) [@RasmussenWilliams:GPs:2006 pp.106-107] i.e.large lengthscales
indicate independence under the covariance matrix for GPs. Examples of
this approach include Hierarchical Diagonal Sampling (HDS)
[@Chen:HDS:2012] and the Dimension Scheduling Algorithm (DSA)
[@Ulmasov:DSA:2016]. The former determines the active variables by a
binary tree of subsets of $\{1, \dots, D\}$, and fits GPs in
lower-dimensional projections. DSA constructs a probability distribution
by the principal directions of the training inputs
$\{(\bm{x}_n, y_n)\}_{n=1}^N$ and subsamples the dimensions accordingly.
In contrast @Li:HDBODropout:2018 randomly sample subsets of active
dimensions.

Other methods rely on placing priors on their lengthscales, followed by
a Bayesian treatment of the training. In Sequential Optimization of
Locally Important Directions (SOLID), lengthscales are weighted by a
Bernoulli distributed parameter, and coordinates are removed when their
posterior probability goes below a user-specified threshold.
[@Munir:SOLID:2021]. @Eriksson:SAASBO:2021 consider the Sparse
Axis-Aligned Subspace (SAAS) model of a GP, restricting the function
space through a (long-tailed) half-Cauchy prior on the
inverse-lengthscales of the kernel.

#### Additive models

Additive models assume that the objective function $f$ can be decomposed
into a sum of lower-dimensional functions. Symbolically, the coordinates
of a given input $\bm{x} = (x_1, \dots, x_D)$ are split into $M$ usually
disjoint subgroups $g_1, \dots g_M$ of smaller size, called a
decomposition. Instead of fitting a GP to $D$ variables in $f$, the
algorithm fits $M$ GPs to the restrictions $f|_{g_1}, \dots f|_{g_M}$
and adds their Upper Confidence Bound. The differences between the
algorithms in this family are on how the subgroups are constructed, how
the additive structure is approximated, the training of the Gaussian
Process, or leveraging special features [@Mutny:HDBOQFF:2018].

@Han:AddGPUCB:2021 select the decomposition which maximizes the marginal
likelihood from a collection of randomly sampled decompositions,
updating it every certain number of iterations. Alternatives include:
leveraging a generalization based on restricted projections
[@Li:RPP:2016], discovering the additive structure using model selection
and Markov Chain Monte Carlo [@Gardner:AddStruct:2017], considering
overlapping groups [@Rolland:GAddGPUCB:2018], ensembles of Mondrian
space-tiling trees [@Wang:BatchedEnsembleHDBO:2018], or use random
tree-based decompositions [@Ziomek:RDUCB:2023].

#### Trust regions

Some BO algorithms restrict the evaluation of the acquisition function
to a small region of input space called a *trust region*, which is
centered at the incumbent and is dynamically contracted or expanded
according to performance
[@Rommel:TRIKE:2016; @Pedrielli:GSTAR:2016; @Eriksson:TuRBO:2019].
Contemporary variants extend to the multivariate setting (e.g.MORBO
[@Daulton:MORBO:2022]), to quality-diversity [@Maus:ROBOT:2023] and to
the optimization of mixed variables (`CASMOPOLITAN` by
@Wan:CASMOPOLITAN:2021), including categorical. Since the trust region
framework involves only the optimization of the acquisition function,
several other methods leverage it alongside other structural assumptions
like linear/non-linear embeddings (e.g.@Tripp:WeightedRetraining:2020
[@Papenmeier:BAxUS:2022]).

#### Linear embeddings {#sec:taxonomy:linear_embeddings}

Instead of optimizing directly in input space $\mathbb{R}^D$, several
methods rely on optimizing in a lower-dimensional space $\mathbb{R}^d$,
which is linearly embedded into data space using a linear transformation
$A\in\mathbb{R}^{D\times d}$ [@Wang:REMBO:2016]. The matrix $A$ can be
either selected at random [@Wang:REMBO:2016; @Qian:SREIMGPO:2016],
computed as a low-rank approximation of the input data matrix
[@Djolonga:HDBandits:2013; @Zhang:SIR:2019; @Raponi:PCABO:2020],
constructed using gradient information and active subspaces
[@Palar:ASM:2017; @Wycoff:AS:2021], or through the minimization of
variance estimates [@Hu:MAVEBO:2024].

These methods are limited by how low-dimensional exploration translates
into high dimensions. One choice of embedding matrix $A$ spans a
*fixed*, highly-restricted subspace of $\mathbb{R}^D$. For this approach
several issues regarding back-projections need to be addressed. Indeed,
projecting from bounded domains $\mathcal{Z}\subseteq\mathbb{R}^d$ to
$\mathbb{R}^D$ might render points outside the bounded domain in the
input [@BinoisWycoff:HDGPs:2022]. Finally, the transformation $A$ is not
injective, meaning a point in input space can correspond to several
latent points [@Binois:WarpedREMBO:2015; @Moriconi:QuantGPBO:2020].

@Binois:WarpedREMBO:2015 propose a kernel that alleviates these issues
by including a back-projection to the bounded domain that respects
distances in the embedded space. *Hashing matrices*
$S\in\mathbb{R}^{D\times d}$ are an alternative way to reconstruct an
input data point in a bounded domain
$\bm{x}\in [-1, 1]^{D}\subseteq\mathbb{R}^D$ from a latent point
$\bm{z}\in\mathbb{R}^D$, whose entries are either 0, 1, and -1. Thus,
the result of multiplying $S\bm{z}$ is a linear combination of the
coordinates of $\bm{z}$ where the coefficients are 1 and -1
[@Nayebi:HESBO:2019]. These ideas have been combined with trust regions
both in the continuous [@Papenmeier:BAxUS:2022] and mixed-variable
settings [@Papenmeier:BOUNCE:2024]. A natural extension considers a
family of nested subspaces, progressively growing the embedding matrix
until it matches the input dimensionality [@Papenmeier:BAxUS:2022]. An
alternative that does not deal with reconstruction mappings (thus
circumventing the aforementioned issues) uses the information learned in
the lower dimensional space to perform optimization directly in input
space [@Horiguchi:MahalaBatchBO:2022].

#### Non-linear embeddings {#sec:taxonomy:non-linear-embeddings}

Several methods have considered non-linear embeddings to incorporate
learned latent representations. One set of examples are deep latent
variable models like Generative Adversarial Networks
[@Goodfellow:GAN:2014], or variants of Autoencoders
[@Kingma:VAE:2014; @Stanton:LAMBO:2022; @Maus:LOLBO:2022], algorithms
that allow for modelling arbitrarily structured inputs. This is highly
relevant for optimizing sequences, which are modeled as samples from a
categorical distribution.

@Bombarelli:AutoChemDesign:2018 pioneered latent space optimization
(LSBO) by learning a latent space of small molecules through their
SMILES representation using a Variational Autoencoder (VAE,
@Kingma:VAE:2014 [@Rezende:VAE:2014]), and optimizing metrics such as
the qualitative estimate of druglikeness (QED) therein. Several
approaches have followed, including usage of *a-priori* given labelled
data [@Eissman:AttrAdjust:2018] or decoder uncertainty
[@Notin:Uncert:2021], smart retraining schemes that focus on promising
points [@Tripp:WeightedRetraining:2020], metric-learning approaches that
match promising points together [@Grosnit:VAEDML:2021], constraining the
latent space [@Griffiths:ConstrainedBOVAEs:2020], latent spaces mapping
to graphs [@Kusner:GrammarVAE:2017; @Jin:JunctionTreeVAE:2018] and
jointly learning the surrogate model and the latent representation
[@Maus:LOLBO:2022; @Lee:CoBo:2023; @Chen:PGLBO:2024; @Kong:DSBO:2024].
@Stanton:LAMBO:2022 take this further by learning multiple
representations: one shared and required for both the decoder and
surrogate, and one discriminative encoding as input for a GP used in the
acquisition function. A prerequisite for these methods is a large
dataset of *unsupervised* inputs, which may not be available in all
applications. The methods that rely on training both the representation
and the regression at the same time need *supervised* labels, which may
be potentially unavailable. Optimization in embedding spaces greatly
increases the complexity of problems that can be tackled, making it an
appealing alternative for discrete sequence optimization in real-world
tasks (see Fig. [2](#fig:overview-of-effective-dim){reference-type="ref"
reference="fig:overview-of-effective-dim"}).

#### Gradient information

High-dimensional problems can become significantly easier when
derivative information is available. Even when the objective's
derivatives are not available, the gradient information from the
surrogate model can guide exploration. In our case, the referenced
approaches cannot be applied directly, as they assume a differentiable
kernel. For methods that rely on a continuous latent representation (see
Secs. [3.4](#sec:taxonomy:linear_embeddings){reference-type="ref"
reference="sec:taxonomy:linear_embeddings"} and
[3.5](#sec:taxonomy:non-linear-embeddings){reference-type="ref"
reference="sec:taxonomy:non-linear-embeddings"}), gradient information
of the surrogate model in latent space can be used.

@Ahmed:FOBO:2016 mention how several Bayesian optimization methods could
leverage gradient information and encourage the community to augment
their optimization schemes with gradients, supported by strong empirical
results even with randomly sampled directional derivatives.
@Eriksson:DSKIP:2018 alleviate the computational constraints that come
from using supervised gradient information using structured kernel
interpolation and computational tricks like fast matrix-vector
multiplication and pivoted Cholesky preconditioning. Other avenues for
mitigating the computational complexity involve using structured
automatic differentiation [@Ament:SBOSDA:2022]. Instead of using the
gradient for taking stochastic steps, @Penubothula:FirstOrder:2021 aim
to find local critical points by querying where the predicted gradient
is zero.

As mentioned above, fitting a Gaussian process to the objective allows
for predicting gradients without having seen them *a priori*
[@RasmussenWilliams:GPs:2006 Sec 9.4]; @Muller:LPSBO:2021 propose
*Gradient Information with BO* (GIBO), in which they guide local policy
search in reinforcement learning tasks, exploiting this property.
@Nguyen:MPD:2022 address that expected gradients may not lead to the
best performing outputs and compute the *most probable descent
direction*.

#### Structured spaces {#sec:taxonomy:structured-spaces}

Some applications work over structured spaces. For example, the angles
of robot arms and protein backbones map to Riemannian manifolds
[@Jaquier:ManifoldsROBO:2020; @Penner:ProtMODULI:2022], and input spaces
might also contain mixed variables (i.e. products of real and continuous
spaces). To compute non-linear embeddings (see
Sec. [3.5](#sec:taxonomy:non-linear-embeddings){reference-type="ref"
reference="sec:taxonomy:non-linear-embeddings"}) followed by standard
Bayesian Optimization (or small variants thereof) can allow us to work
over such spaces. @Jaquier:GABO:2020 use kernels defined on Riemannian
manifolds
[@Feragen:GeodesicKernels:2015; @Borovitsky:MaternKernelOnManifold:2020]
and optimize the acquisition function using tools from Riemannian
optimization [@Boumal:ManifoldOpt:2023]. The authors expand their
framework to high-dimensional manifolds by projecting to
lower-dimensional submanifolds, which is roughly the equivalent to
*linear embeddings* in the Riemannian settings [@Jaquier:HDGABO:2020].

In the categorical and mixed-variable setting, kernels over string
spaces [@Lodhi:SSKs:2000; @Shervashidze:WLGraphKernel:2011], can be
applied to BO [@Moss:BOSS:2020]. Other methods construct combinatorial
graph and diffusion kernels-based GPs [@Oh:COMBO:2019].
@Deshwal:LADDER:2021 combine latent space kernels with combinatorial
kernels in an autoencoder-based approach.

Recently, @Daulton:PR:2022 have proposed a continuous relaxation of the
discrete variables to ease the optimization of the acquisition function.
@Deshwal:BODI:2023 propose another way to map discrete variables to
continuous space, relying on Hamming distances to make a dictionary for
embeddings. @Papenmeier:BOUNCE:2024 extend previous work to both
continuous and categorical variables: `BAxUS` learns an increasing
sequence of subspaces using hash matrices which, when combined with the
`CoCaBo` kernel [@Ru:CoCaBO:2020], renders an algorithm for the
mixed-variable setting. Finally, through a continuous relaxation of the
objective that incorporates *prior* pretrained models,
@Michael:COREL:2024 propose a surrogate on the probability vector space
to optimize either the discrete input space or a continuous latent one.

**Other.**   Some methods evade our taxonomy but are worth mentioning:
some focus on the optimization of the acquisition function and the
impact of initializations [@Zhao:AIBO:2024; @Ngo:CMABO:2024]. Other
methods balance both active learning (i.e.building a better surrogate
model) and optimization [@Hvarfner:SCOREBO:2023]. Most recently, two
articles claimed that the standard setting for Bayesian optimization or
slight variations of it perform as well as the state-of-the-art of all
the aforementioned families
[@Hvarfner:VanilaBO:2024; @Xu:StdVanillaBO:2024] -- begging the
question, can these methods optimize in high dimensional discrete
problem spaces in a sample efficient manner?