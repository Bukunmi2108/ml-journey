# 🚀 ML/AI/Deep Learning — 30-Week Actionable Timeline
## Bukunmi Akinyemi | Start Date: 22-02-2026 | Target End: _________

---

## HOW TO USE THIS DOCUMENT

- **Each day has a specific task with clear explanation of what to do and why.**
- **⏰ = estimated hours** for that day's work
- **✅ = deliverable** you must complete before moving on. If you can't show it, you didn't do it.
- **Weekdays** = study + implement. **Saturdays** = project days. **Sundays** = review + consolidate.
- **Daily rhythm:** Morning = study/lectures. Afternoon/evening = code. Night = commit to GitHub + post on X.
- **If you fall behind:** Never skip two days in a row. Compress the next rest day. Never delete tasks — push them forward.

---

## THE ACCOUNTABILITY SYSTEM

Past projects didn't stick because there was no external structure enforcing consistency. This plan fixes that with four mechanisms:

**1. Daily GitHub commits.** Every single day, something gets pushed. Even if it's just notes. Your GitHub contribution graph is your visual proof of consistency. A green square every day.

**2. Daily X posts.** You've committed to posting daily progress. This creates social accountability — people will notice if you stop. Keep posts short: what you did, one thing you learned.

**3. Weekly checkpoints.** Every Sunday you review the week and update the progress tracker at the bottom. If you're behind, you adjust the plan — you don't abandon it.

**4. The 48-hour rule.** If you miss one day, that's rest. If you're about to miss a second consecutive day, do ANYTHING — even 30 minutes of review or one LeetCode problem. Two missed days is where every abandoned learning plan dies. Break the chain of inaction, not the chain of progress.

---

## SETUP DAY (Day 0 — Before You Start)

**Complete everything below in one sitting. Estimated time: 2 hours. Do not start Day 1 until this is done.**

- [x] Create a GitHub repo called `ml-journey`. Initialize with a README that states your goal and timeline.
- [x] Set up Google Colab — go to colab.research.google.com, sign in, create a test notebook, verify you can access a GPU (Runtime → Change runtime type → T4 GPU).
- [x] Create a Kaggle account at kaggle.com. Verify your phone number — this is required to access free GPU quotas (30 hours/week of T4 GPU). Go to kaggle.com/learn to preview the micro-courses.
- [ ] Install locally: Python 3.10+, VS Code, Git. Run `pip install torch torchvision numpy pandas matplotlib scikit-learn jupyter` to verify your environment works.
- [ ] Create a Weights & Biases account at wandb.ai (free tier). You'll use this later for experiment tracking.
- [ ] Create a folder structure in your repo: `/notebooks`, `/projects`, `/notes`, `/implementations`
- [ ] Bookmark these URLs — you'll use them constantly:
  - karpathy.ai/zero-to-hero.html
  - course.fast.ai
  - coursera.org/specializations/machine-learning-introduction
  - kaggle.com/learn
  - huggingface.co/learn
  - introtodeeplearning.com
  - cs231n.github.io
- [ ] Post your Day 0 announcement on X.

---

# ═══════════════════════════════════════════
# PHASE 0 + 1: MATH & PYTHON FOUNDATIONS
# Weeks 1–3 (Days 1–21)
# ═══════════════════════════════════════════

The reason most people fail at ML is not intelligence — it's that they skip foundations and then hit a wall at backpropagation or gradient descent. These three weeks prevent that wall. Every hour here saves you ten hours of confusion later.

---

## WEEK 1: Linear Algebra + Python
**Why this matters:** Every ML model is a sequence of matrix operations. If you can't think in vectors and matrices, neural networks will always feel like black boxes.

### Day 1 (Monday) ⏰ 3hrs
**Linear Algebra — Vectors and Spaces**

Watch 3Blue1Brown "Essence of Linear Algebra" videos 1–4 on YouTube. These cover vectors, linear combinations, spans, and matrix-vector multiplication. The key insight you need from today: a matrix is not just a grid of numbers — it's a transformation that moves space around.

After watching each video, open a Google Colab notebook and implement what you just saw. Create vectors as NumPy arrays. Multiply them. Visualize 2D transformations by plotting how a matrix moves a grid of points. Use `matplotlib` to plot before-and-after transformations.

By end of day, you should be able to explain in your own words what a linear transformation is and demonstrate it with code.

- [ ] ✅ Notebook: vector operations, dot products, matrix-vector multiplication, 2D transformation visualization — pushed to GitHub

### Day 2 (Tuesday) ⏰ 3hrs
**Linear Algebra — Matrix Operations**

Watch 3Blue1Brown videos 5–8: determinants, inverse matrices, column space/null space, and nonsquare matrices. The determinant tells you how much a transformation scales area — if it's zero, the transformation crushes space into a lower dimension (and the matrix has no inverse). This concept comes back in PCA and regularization.

Implement in Colab: compute determinants with `np.linalg.det()`, find inverses with `np.linalg.inv()`, solve the system Ax = b using both `np.linalg.solve()` and by manually computing A⁻¹b. Then try a singular matrix (determinant = 0) and observe what happens when you try to invert it.

- [ ] ✅ Notebook: determinants, matrix inverses, solving linear systems, singular matrix demonstration — pushed to GitHub

### Day 3 (Wednesday) ⏰ 3hrs
**Linear Algebra — Eigenvalues and Eigenvectors**

Watch 3Blue1Brown videos 9–13: dot products as projections, eigenvectors and eigenvalues. This is the most important concept for today: an eigenvector of a matrix is a direction that doesn't rotate under the transformation — it only gets scaled. The scaling factor is the eigenvalue.

Why this matters for ML: Principal Component Analysis (PCA) finds the eigenvectors of the covariance matrix to identify the directions of maximum variance in data. This is how dimensionality reduction works.

Implement: use `np.linalg.eig()` to find eigenvalues and eigenvectors. Create a 2x2 matrix, compute its eigenvectors, then apply the matrix to those eigenvectors and verify they only get scaled. Visualize this. Then apply PCA to a simple 2D dataset and show how the first principal component aligns with the dominant eigenvector.

- [ ] ✅ Notebook: eigendecomposition, eigenvector visualization, simple PCA demonstration — pushed to GitHub

### Day 4 (Thursday) ⏰ 3hrs
**Python Proficiency — Kaggle Python Course**

Complete the Kaggle Learn "Python" micro-course at kaggle.com/learn/python. This covers: functions, booleans, conditionals, lists, loops, strings, dictionaries, and working with external libraries. Even if your Python is decent, the exercises force you to use specific patterns that come up constantly in ML code.

Work through every exercise, not just the ones that seem hard. The goal is speed and fluency — you should be able to write Python without thinking about syntax so you can focus on algorithms later.

Also complete 3Blue1Brown videos 14–16 (abstract vector spaces) if you haven't finished the series.

- [ ] ✅ Kaggle Python course 100% complete with certificate earned
- [ ] ✅ 3Blue1Brown Linear Algebra series complete

### Day 5 (Friday) ⏰ 3hrs
**Data Manipulation — Pandas**

Start and ideally complete the Kaggle Learn "Pandas" micro-course. Pandas is the standard tool for data manipulation in Python and you'll use it in every single ML project. The course covers: creating DataFrames, indexing/selecting, summary functions, grouping, data types, missing values, renaming, and combining DataFrames.

Pay special attention to `.groupby()`, `.merge()`, and handling missing values (`.fillna()`, `.dropna()`) — these three operations account for most of the data preprocessing work in real ML projects.

Practice beyond the exercises: download any CSV from Kaggle Datasets, load it, and perform 10 different operations on it.

- [ ] ✅ Kaggle Pandas course complete with certificate
- [ ] ✅ Additional practice notebook with a real dataset — pushed to GitHub

### Day 6 (Saturday — Project Day) ⏰ 4-5hrs
**Data Visualization + First EDA Project**

Morning: Complete the Kaggle Learn "Data Visualization" micro-course. This covers seaborn: line charts, bar charts, heatmaps, scatter plots, histograms, and density plots.

Afternoon: Your first real mini-project. Go to kaggle.com/datasets, find a dataset that interests you (anything — sports, finance, health, music). Download it. In a fresh Colab notebook:
1. Load the data with pandas
2. Examine shape, dtypes, missing values, basic statistics
3. Create at minimum 5 different visualizations that tell a story about the data
4. Write markdown cells explaining what each visualization reveals
5. Push the complete notebook to your GitHub repo under `/projects/week1-eda/`

This is your first portfolio piece. Make it clean and readable.

- [ ] ✅ Data Visualization certificate earned
- [ ] ✅ Complete EDA notebook on a real dataset with 5+ visualizations — on GitHub

### Day 7 (Sunday — Review + Consolidate) ⏰ 2hrs
**Weekly Review**

Go through every notebook you created this week. For each one, write a 2-3 sentence summary at the top explaining what it demonstrates. Update your repo README with a Week 1 section listing what you completed.

Then test yourself: close all notebooks and answer these questions on paper or in a new markdown file:
- What is a linear transformation?
- What does the determinant of a matrix tell you?
- What are eigenvectors and why do they matter for ML?
- How do you solve Ax = b in NumPy?
- What does `.groupby()` do in Pandas?

If you can't answer any of these clearly, revisit that day's material.

- [ ] ✅ All notebooks annotated
- [ ] ✅ README updated with Week 1 summary
- [ ] ✅ Self-test completed honestly

---

## WEEK 2: Calculus, Probability & Statistics
**Why this matters:** Calculus gives you gradients (how neural networks learn). Probability gives you loss functions and evaluation metrics (how you measure whether learning worked). Without these, training a model is just pressing buttons without understanding what's happening.

### Day 8 (Monday) ⏰ 3hrs
**Calculus — Derivatives and the Chain Rule**

Watch 3Blue1Brown "Essence of Calculus" videos 1–5. Focus heavily on: what a derivative actually means (instantaneous rate of change), the power rule, the chain rule. The chain rule is the mathematical backbone of backpropagation — when you train a neural network, you're computing the chain rule through thousands of operations.

Implement in a notebook: write a function `numerical_derivative(f, x, h=1e-7)` that approximates the derivative of any function f at point x. Test it on f(x) = x², f(x) = sin(x), f(x) = e^x. Compare your numerical answers to the analytical answers. Then implement the chain rule: for f(g(x)), show that df/dx = df/dg · dg/dx numerically.

This seems simple. It is. And it's exactly what PyTorch's autograd does at scale.

- [ ] ✅ Notebook: numerical derivatives, chain rule verification — pushed to GitHub

### Day 9 (Tuesday) ⏰ 3hrs
**Calculus — Gradients and Neural Network Preview**

Watch 3Blue1Brown Calculus videos 6–10 (integrals, higher dimensions). Then watch the 3Blue1Brown "Neural Networks" series — all 4 videos. Don't implement anything from the neural network videos yet. Just absorb the visual intuition: networks as function composition, gradient descent as rolling downhill, backpropagation as the chain rule applied backwards through layers.

Write detailed notes connecting the calculus concepts to neural network training. Specifically: how does computing df/dx (a simple derivative) connect to computing dLoss/dWeight (what backpropagation computes)?

Then implement gradient descent from scratch on a simple function: find the minimum of f(x) = (x - 3)² + 2 by starting at a random x and iteratively stepping in the direction of -df/dx. Visualize the path it takes to the minimum.

- [ ] ✅ Written notes: "How calculus connects to neural networks" (at least 1 page)
- [ ] ✅ Notebook: gradient descent on a simple function with visualization — pushed to GitHub

### Day 10 (Wednesday) ⏰ 3hrs
**Probability — Distributions and Bayes' Theorem**

Watch StatQuest videos on: Probability basics, Bayes' Theorem, Normal/Gaussian distribution, Bernoulli distribution, Binomial distribution. Josh Starmer explains these with extreme clarity — don't skip the "clearly explained" parts even if they seem basic.

Implement: write a simulation that flips a biased coin 10,000 times and plots the resulting distribution. Show how it converges to the expected probability. Implement Bayes' theorem as a function and solve a classic problem (e.g., "given a positive test result, what's the probability you actually have the disease?"). Plot Normal distributions with different means and standard deviations using matplotlib.

Why this matters: every classification model outputs probabilities. Every loss function is derived from probability theory. You need this to be intuitive, not memorized.

- [ ] ✅ Notebook: probability simulations, Bayes' theorem, distribution visualizations — pushed to GitHub

### Day 11 (Thursday) ⏰ 3hrs
**Loss Functions and Information Theory**

Watch StatQuest videos on: Maximum Likelihood Estimation, Cross-Entropy, Entropy, KL Divergence. These are dense but critical — cross-entropy is THE standard loss function for classification in neural networks, and understanding WHY requires information theory.

Implement from scratch in Python:
1. `binary_cross_entropy(y_true, y_pred)` — compute by hand using the formula, not a library
2. `categorical_cross_entropy(y_true, y_pred)` — for multi-class
3. `mean_squared_error(y_true, y_pred)` — for regression
4. Show numerically that cross-entropy is minimized when predictions match true labels
5. Plot how cross-entropy loss changes as your prediction moves away from the true label

After implementing these, you should be able to answer: "Why do we use cross-entropy instead of accuracy as a loss function?" If you can't, re-watch the StatQuest videos.

- [ ] ✅ Notebook: loss functions implemented from scratch with visualizations and explanations — pushed to GitHub

### Day 12 (Friday) ⏰ 3hrs
**Kaggle ML Courses Begin**

Complete the Kaggle Learn "Intro to Machine Learning" micro-course. This is your first hands-on ML course — covers: how models work, basic data exploration, first model (decision trees), model validation, underfitting/overfitting, and random forests.

The exercises use the Melbourne Housing dataset. Pay careful attention to the train/validation split concept — this is where most beginners make their first mistake (evaluating on training data and thinking their model is amazing).

After completing the course exercises, extend the final notebook: try at least 3 different values of `max_leaf_nodes` for the decision tree and plot the validation MAE for each. Find the optimal depth.

- [ ] ✅ Kaggle Intro to ML certificate earned
- [ ] ✅ Extended notebook with hyperparameter exploration — pushed to GitHub

### Day 13 (Saturday — Project Day) ⏰ 4-5hrs
**Intermediate ML + First Serious Preprocessing**

Complete the Kaggle Learn "Intermediate Machine Learning" course. This covers the practical skills that separate toy ML from real ML: handling missing values (3 strategies), categorical variables (ordinal encoding, one-hot encoding), pipelines for clean code, cross-validation for reliable evaluation, and XGBoost (the most powerful algorithm for tabular data).

XGBoost deserves special attention. It wins most Kaggle tabular competitions. Understand what gradient boosting means conceptually: you train a sequence of weak models where each new model focuses on correcting the errors of the previous ones.

After completing the course, try XGBoost on the Melbourne Housing dataset and compare its performance to the random forest from yesterday.

- [ ] ✅ Kaggle Intermediate ML certificate earned
- [ ] ✅ XGBoost comparison notebook — pushed to GitHub

### Day 14 (Sunday — Review + Consolidate) ⏰ 2hrs
**Weekly Review + DSA Start**

Review all Week 2 notebooks. Self-test:
- What is the chain rule and why does it matter for neural networks?
- Write the binary cross-entropy formula from memory.
- What's the difference between underfitting and overfitting?
- When should you use ordinal encoding vs. one-hot encoding?

Start your LeetCode practice: create an account, solve 2 Easy problems from the "Arrays" category. DSA practice continues in parallel from here — minimum 2 problems per week, ideally 3-5.

- [ ] ✅ Self-test complete
- [ ] ✅ 2 LeetCode Easy problems solved
- [ ] ✅ Week 2 README update

---

## WEEK 3: DSA Foundation + First Kaggle Competition
**Why this matters:** Algorithms teach you to think about efficiency and tradeoffs — skills that transfer directly to ML system design. The Kaggle competition gives you your first end-to-end ML experience.

### Day 15 (Monday) ⏰ 3hrs
**Algorithms — Divide and Conquer**

Start the Stanford Algorithms Specialization on Coursera (audit free) — Course 1 by Tim Roughgarden. Watch Week 1 lectures: Introduction, Merge Sort analysis, asymptotic notation (Big-O).

Implement merge sort from scratch in Python. Do not look at any implementation while coding — write it from the algorithm description. Time your implementation on arrays of size 1,000, 10,000, and 100,000 and verify it scales as O(n log n).

Then implement counting inversions (a variation of merge sort) — this is Week 1's programming assignment.

- [ ] ✅ Merge sort from scratch + timing analysis
- [ ] ✅ Counting inversions implementation
- [ ] ✅ Both pushed to GitHub under `/implementations/sorting/`

### Day 16 (Tuesday) ⏰ 3hrs
**Algorithms — QuickSort + Randomization**

Stanford Algorithms Course 1, Week 2: QuickSort and its analysis, the importance of random pivot selection. QuickSort is expected O(n log n) but worst-case O(n²) — the randomization makes the worst case astronomically unlikely.

Implement quicksort with three pivot strategies: (1) always pick first element, (2) always pick last element, (3) random pivot. Run all three on a sorted array of 100,000 elements and time them. The first two will be visibly slower — this demonstrates why randomization matters.

LeetCode: solve 2 problems (one array, one string).

- [ ] ✅ Quicksort with 3 pivot strategies + benchmarks
- [ ] ✅ 2 LeetCode problems solved

### Day 17 (Wednesday) ⏰ 3hrs
**Algorithms — Graph Basics + BFS/DFS**

Stanford Algorithms Course 1, Week 3: Graph representations, Breadth-First Search, Depth-First Search, topological sort, connected components.

Implement in Python: represent a graph as an adjacency list (dictionary of lists). Implement BFS and DFS both iteratively and recursively. Find connected components in an undirected graph. Apply topological sort to a directed acyclic graph.

These graph algorithms come back in neural network computation graphs, knowledge graphs for RAG systems, and dependency analysis.

LeetCode: 2 graph/tree problems (Easy).

- [ ] ✅ Graph representations + BFS + DFS + topological sort implementations
- [ ] ✅ 2 LeetCode problems

### Day 18 (Thursday) ⏰ 3hrs
**Algorithms — Shortest Paths**

Stanford Algorithms Course 1, Week 4: Dijkstra's algorithm for shortest paths. This is one of the most important algorithms in computer science.

Implement Dijkstra's using a min-heap (Python's `heapq`). Test on a weighted graph you construct by hand. Visualize the shortest path.

Then implement binary search from scratch — this doesn't come from the course but is fundamental. Write it iteratively and recursively. Make sure you handle edge cases (empty array, element not found, duplicates).

LeetCode: 2 binary search problems.

- [ ] ✅ Dijkstra's algorithm implementation
- [ ] ✅ Binary search (iterative + recursive)
- [ ] ✅ 2 LeetCode problems

### Day 19 (Friday) ⏰ 3hrs
**ML Practice — Kaggle Titanic Competition Prep**

Today you prepare for tomorrow's project by studying the Titanic dataset and reading top notebooks on Kaggle. Go to kaggle.com/c/titanic. Read the data description thoroughly. Download the data and explore it in a notebook:
- What features exist?
- What's the distribution of survival?
- How much data is missing and where?
- What feature engineering ideas come to mind?

Read 2-3 highly-upvoted Kaggle notebooks for the Titanic competition. Don't copy their code — study their thought process. Note what feature engineering techniques they use (title extraction from names, family size, age binning, cabin deck).

Plan your approach for tomorrow.

- [ ] ✅ Titanic EDA notebook with feature analysis
- [ ] ✅ Written plan for tomorrow's competition submission

### Day 20 (Saturday — Major Project Day) ⏰ 5-6hrs
**FIRST KAGGLE COMPETITION: Titanic Survival Prediction**

This is your first end-to-end ML project. Execute the full pipeline without tutorials:

1. **Data loading and EDA** — load train.csv and test.csv, examine distributions, visualize survival rates by feature
2. **Feature engineering** — extract titles from names, create family_size, bin ages, encode deck from cabin, create is_alone feature
3. **Handle missing data** — impute age with median by title, fill embarked with mode, drop cabin (too many missing)
4. **Encode categoricals** — one-hot or label encode Sex, Embarked, Title
5. **Model training** — train at least 4 models: Logistic Regression, Random Forest, XGBoost, and one more of your choice
6. **Evaluation** — cross-validation scores for each model, confusion matrix for the best one
7. **Submission** — generate predictions on test set, submit to Kaggle, record your score
8. **Documentation** — clean up the notebook with markdown explanations throughout

Push the complete notebook to GitHub. This is a portfolio piece — make it clean.

- [ ] ✅ Kaggle Titanic submission made — Score: ______
- [ ] ✅ Complete, documented notebook on GitHub
- [ ] ✅ At least 4 models compared with cross-validation

### Day 21 (Sunday — Review + Milestone) ⏰ 2hrs
**3-Week Milestone Review**

You have now completed: linear algebra, calculus, probability/statistics, 5 Kaggle micro-course certificates, sorting and graph algorithms, 10+ LeetCode problems, and your first Kaggle competition.

Self-assessment — answer honestly:
- Can I implement gradient descent from scratch? (If no → revisit Day 9)
- Can I explain cross-entropy loss? (If no → revisit Day 11)
- Can I build a complete ML pipeline from data to submission? (If no → revisit Day 20)
- Can I implement merge sort and binary search without references? (If no → revisit Days 15-18)

Write a summary X thread covering your first 3 weeks. Update your GitHub README comprehensively.

- [ ] ✅ Self-assessment completed
- [ ] ✅ GitHub repo fully organized with clear READMEs
- [ ] ✅ 3-week milestone thread posted on X

---

# ═══════════════════════════════════════════
# PHASE 3: MACHINE LEARNING MASTERY
# Weeks 4–6 (Days 22–42)
# ═══════════════════════════════════════════

These three weeks transform you from "knows some ML" to "can build and reason about ML systems." You'll complete Andrew Ng's ML Specialization AND start Karpathy's neural networks series.

---

## WEEK 4: Ng ML Specialization — Supervised Learning

### Day 22 (Monday) ⏰ 3hrs
**Ng Course 1, Week 1: Linear Regression + Gradient Descent**

Start Andrew Ng's Machine Learning Specialization on Coursera (audit free). Course 1: "Supervised Machine Learning: Regression and Classification." Watch all Week 1 videos. Ng explains linear regression, cost functions, and gradient descent with exceptional clarity.

Complete all optional labs — they're in Jupyter notebooks using Python. Even if you already implemented gradient descent in Week 2, Ng frames it differently and the repetition from a new angle deepens understanding.

- [ ] ✅ All Week 1 videos watched + labs complete

### Day 23 (Tuesday) ⏰ 3-4hrs
**Implement Linear Regression — No Libraries**

Close Ng's labs. Open a blank notebook. Implement the complete linear regression pipeline from scratch:
- Generate synthetic data: y = 3x + 7 + noise
- Implement the cost function (MSE) as a Python function
- Implement gradient descent — compute partial derivatives of cost w.r.t. slope and intercept
- Run gradient descent for 1000 iterations, storing the cost at each step
- Plot: (1) the data + regression line, (2) cost vs. iteration number
- Extend to multiple features: implement multivariate linear regression using matrix operations

The purpose of implementing from scratch isn't to avoid scikit-learn — it's to understand what scikit-learn is doing internally so you can debug when things go wrong.

LeetCode: 2 problems (hash maps).

- [ ] ✅ Linear regression from scratch (single + multivariate) — pushed to GitHub
- [ ] ✅ 2 LeetCode problems

### Day 24 (Wednesday) ⏰ 3hrs
**Ng Course 1, Week 2: Multiple Features + Feature Scaling**

Watch Week 2 videos + complete labs. Key concepts: vectorized implementation (using NumPy instead of loops), feature scaling (normalization and standardization), polynomial features, learning rate selection.

After the labs, experiment with learning rates: run gradient descent with α = 0.001, 0.01, 0.1, 1.0 on the same dataset. Plot convergence curves for each. This builds intuition for hyperparameter tuning that you'll use throughout your career.

- [ ] ✅ Week 2 videos + labs complete
- [ ] ✅ Learning rate experiment notebook

### Day 25 (Thursday) ⏰ 3hrs
**Ng Course 1, Week 3: Classification + Logistic Regression**

Watch Week 3 videos + labs. Logistic regression introduces the sigmoid function, binary cross-entropy loss, and the concept of a decision boundary. Ng also covers regularization (L1 and L2) — the technique of penalizing model complexity to prevent overfitting.

Pay special attention to the regularization section. Understand intuitively: L2 regularization pushes weights toward zero (but not exactly zero), while L1 regularization can push weights to exactly zero (creating sparse models). This matters for feature selection and model interpretability.

- [ ] ✅ Week 3 videos + labs complete + quiz passed

### Day 26 (Friday) ⏰ 3-4hrs
**Implement Logistic Regression — No Libraries**

Same drill as Day 23 but for classification:
- Generate a 2D binary classification dataset (use `sklearn.datasets.make_classification` for the data, but implement the model yourself)
- Implement the sigmoid function
- Implement binary cross-entropy loss
- Implement gradient descent for logistic regression
- Plot the decision boundary
- Add L2 regularization and show how it affects the decision boundary

Then compare your from-scratch implementation with `sklearn.linear_model.LogisticRegression` on the same data. The accuracy should be nearly identical.

LeetCode: 2 problems (stacks/queues).

- [ ] ✅ Logistic regression from scratch with regularization — pushed to GitHub
- [ ] ✅ Comparison with scikit-learn implementation

### Day 27 (Saturday — Project Day) ⏰ 5-6hrs
**Kaggle House Prices Competition**

Your second Kaggle competition — significantly harder than Titanic because it's regression with many features, extensive missing data, and requires serious feature engineering.

Go to kaggle.com/c/house-prices-advanced-regression-techniques. Execute:
1. **Deep EDA** — analyze all 79 features, identify their types, missing value patterns, correlations with SalePrice
2. **Feature engineering** — create TotalSF (total square footage), TotalBath, age-related features, quality interactions
3. **Preprocessing** — handle missing values intelligently (not just median fill), encode categoricals, handle skewed distributions with log transforms
4. **Model comparison** — train at minimum: Linear Regression, Ridge, Lasso, Random Forest, XGBoost, LightGBM
5. **Ensemble** — average top 2-3 models' predictions
6. **Submit** and record your score

Aim for a score in the top 50% on your first attempt. You can iterate and improve later.

- [ ] ✅ Kaggle House Prices submission — Score: ______
- [ ] ✅ Complete pipeline notebook with 6+ models — pushed to GitHub

### Day 28 (Sunday) ⏰ 2hrs
**Week 4 Review**

Review notebooks. Update README. Self-test: implement gradient descent for linear regression on paper (no code). If you can write the weight update equations from memory, your foundations are solid.

LeetCode: 1 Medium problem.

- [ ] ✅ Self-test passed
- [ ] ✅ Week 4 README update

---

## WEEK 5: Neural Networks Begin — Ng + Karpathy

This is the pivotal week. Everything before was building to this. Everything after builds on this.

### Day 29 (Monday) ⏰ 3hrs
**Ng Course 2, Week 1: Neural Network Fundamentals**

Start "Advanced Learning Algorithms" (Course 2). Week 1 covers: neurons, layers, activation functions (sigmoid, ReLU, linear), forward propagation. Ng builds up from a single neuron to a multi-layer network.

Complete all labs. In the TensorFlow lab, pay attention to the syntax but understand that you're about to build this from scratch with Karpathy — the library is doing exactly what you'll implement by hand.

Key insight to internalize: a neural network is just a composition of simple functions. Layer 1 output feeds into Layer 2 input. That's it. The magic is in learning the weights through backpropagation.

- [ ] ✅ Course 2 Week 1 complete (videos + labs + quiz)

### Day 30 (Tuesday) ⏰ 3hrs
**Ng Course 2, Weeks 2-3: Backpropagation + Practical Advice**

Week 2: How neural networks learn — backpropagation, activation function choices (ReLU > sigmoid for hidden layers), multiclass classification with softmax. Week 3: bias/variance, regularization strategies, iterative ML development, error analysis.

The Week 3 content on practical ML advice is underrated. Ng's framework for "what to try next when your model isn't working" will save you hundreds of hours over your career. The key decision tree: high bias → bigger network or more features. High variance → more data or regularization.

- [ ] ✅ Course 2 Weeks 2-3 complete

### Day 31 (Wednesday) ⏰ 4hrs — THE MOST IMPORTANT DAY
**KARPATHY: Building micrograd — Autograd from Scratch**

Watch Andrej Karpathy's "The spelled-out intro to neural networks and backpropagation: building micrograd" (youtube — approximately 2.5 hours). Code along in a Colab notebook as he builds.

What you're building: a tiny automatic differentiation engine (like PyTorch's autograd) in pure Python. It supports:
- A `Value` class that wraps numbers and tracks computational graphs
- Forward pass: normal arithmetic operations
- Backward pass: automatic gradient computation using the chain rule

By the end of this lecture, you will have built a working autograd engine that can train a neural network — in about 100 lines of Python. This is the foundational mechanism behind ALL modern deep learning.

Do not rush this. Pause the video when needed. Make sure you understand every line. This is the single most important lecture in the entire 30-week plan.

- [ ] ✅ Complete micrograd implementation coded along with Karpathy — pushed to GitHub

### Day 32 (Thursday) ⏰ 3-4hrs
**Rebuild micrograd from MEMORY**

Close Karpathy's video. Close your notebook from yesterday. Open a completely blank file. Rebuild micrograd from scratch.

You WILL get stuck. That's the point. Every place you get stuck reveals a gap in understanding. When stuck:
1. Think for 5 minutes
2. If truly stuck, peek at your Day 31 notebook for the smallest possible hint
3. Close it immediately and continue

Areas where people commonly get stuck: (1) implementing the backward pass for division/subtraction, (2) getting the chain rule accumulation right (+=, not =), (3) the topological sort for the backward pass order.

After rebuilding, test your engine by training a small neural network on a toy dataset (same as Karpathy does).

LeetCode: 2 problems (linked lists).

- [ ] ✅ Micrograd rebuilt from memory (or near-memory) — this version pushed to GitHub separately

### Day 33 (Friday) ⏰ 3hrs
**Ng Course 2, Week 4: Decision Trees + Ensemble Methods**

Complete the final week of Course 2: decision trees, random forests, XGBoost, when to use neural networks vs. decision trees. Ng's practical advice: for tabular/structured data, tree-based methods (especially XGBoost) often outperform neural networks. Neural networks shine on unstructured data (images, text, audio).

Then start Course 3: "Unsupervised Learning, Recommenders and Reinforcement Learning." Complete Week 1: K-means clustering and anomaly detection.

- [ ] ✅ Course 2 complete
- [ ] ✅ Course 3 Week 1 complete

### Day 34 (Saturday — Project Day) ⏰ 5hrs
**KARPATHY: Building makemore Part 1 — Bigram Language Model**

Watch and code along with "The spelled-out intro to language modeling: building makemore." This builds a character-level language model that generates names. It starts with pure counting statistics (bigram model) and then shows how to express the same thing as a neural network.

The key insight: a bigram model says "given the previous character, what's the probability of the next character?" This is the simplest possible language model. Everything from here to GPT-4 is variations on this same question with longer context.

After completing the lecture: try it with a different dataset. Download a list of words (city names, species names, whatever interests you) and train the bigram model on that. Observe what it generates.

- [ ] ✅ Bigram language model complete + tested on custom dataset — pushed to GitHub

### Day 35 (Sunday) ⏰ 2hrs
**Week 5 Review + Course 3 Completion**

Complete Ng Course 3 Weeks 2-3 (recommender systems and intro to RL). This finishes the entire ML Specialization.

Review your micrograd and makemore implementations. Update GitHub.

- [ ] ✅ Andrew Ng ML Specialization — ALL 3 COURSES COMPLETE ✅
- [ ] ✅ Week 5 summary posted on X

---

## WEEK 6: Karpathy Makemore Series + Deep Neural Network Practice

### Day 36 (Monday) ⏰ 3-4hrs
**KARPATHY: makemore Part 2 — MLP Language Model**

Watch and code along with "Building makemore Part 2: MLP." This upgrades from the bigram model to a multi-layer perceptron that looks at multiple previous characters. You'll implement Bengio et al.'s 2003 neural language model from scratch.

Key concepts: character embeddings (turning characters into vectors), hidden layers, training loops, train/validation/test splits for language models. The embedding concept is crucial — it comes back in BERT, GPT, and every modern NLP model.

- [ ] ✅ MLP language model complete — pushed to GitHub

### Day 37 (Tuesday) ⏰ 3-4hrs
**KARPATHY: makemore Part 3 — Activations and Gradients**

Watch and code along with "Building makemore Part 3: Activations & Gradients, BatchNorm." This is Karpathy's lecture on the craft of training neural networks — diagnosing problems by inspecting activations, gradients, and weight distributions.

Key skills: visualizing activation distributions per layer (are neurons dying?), gradient flow visualization (are gradients vanishing or exploding?), how BatchNorm fixes training stability, proper weight initialization (Kaiming init).

These diagnostic skills separate people who can tune models from people who just run default settings and hope for the best.

- [ ] ✅ Activations and gradients diagnostic notebook — pushed to GitHub

### Day 38 (Wednesday) ⏰ 3-4hrs
**KARPATHY: makemore Part 4 — Becoming a Backprop Ninja**

Watch and code along with "Building makemore Part 4: Becoming a backprop ninja." This is the hardest Karpathy lecture. You will manually compute the backward pass through every operation in the network: cross-entropy, softmax, batch normalization, tanh, matrix multiply, embedding lookup.

This is painful. It is also the exercise that makes backpropagation truly yours. After this, you will never be confused about how gradients flow through a network.

If you get completely stuck on a derivation, it's okay to watch Karpathy's solution — but try each one yourself for at least 10 minutes first.

- [ ] ✅ Manual backprop through all operations — pushed to GitHub

### Day 39 (Thursday) ⏰ 3-4hrs
**KARPATHY: makemore Part 5 — WaveNet Architecture**

Watch and code along with "Building makemore Part 5: Building a WaveNet." This introduces dilated causal convolutions — a more sophisticated architecture that can look at longer contexts efficiently.

After completing this, the entire makemore series is done. You've gone from counting character pairs to training a WaveNet. Summarize in your GitHub README what each part taught you.

LeetCode: 2 problems (trees/graphs).

- [ ] ✅ WaveNet implementation complete
- [ ] ✅ Makemore series complete — all 5 parts on GitHub

### Day 40 (Friday) ⏰ 3hrs
**Start Ng Deep Learning Specialization**

Begin Andrew Ng's Deep Learning Specialization on Coursera (5 courses). Start Course 1: "Neural Networks and Deep Learning." Complete Weeks 1-2: Introduction, logistic regression as a neural network, shallow neural networks.

Much of this will be review given the Karpathy work — that's intentional. Ng provides the mathematical formalism and notation that Karpathy teaches through code. Having both perspectives solidifies understanding.

- [ ] ✅ DL Spec Course 1 Weeks 1-2 complete

### Day 41 (Saturday — Project Day) ⏰ 5-6hrs
**Customer Churn Prediction + DL Spec Progress**

Morning: Complete DL Spec Course 1 Weeks 3-4 (deep neural networks, hyperparameters, matrix dimensions). Finish Course 1.

Afternoon: Build a Customer Churn Prediction project. Find a churn dataset on Kaggle (e.g., Telco Customer Churn). This is a business-oriented classification problem.

The twist: this time, focus on evaluation metrics beyond accuracy. For churn prediction, a false negative (missing a customer who will churn) is much worse than a false positive. Use precision, recall, F1-score, ROC-AUC, and precision-recall curves. Tune the classification threshold for business impact.

Compare: Logistic Regression, Random Forest, XGBoost, and a simple neural network (using PyTorch or Keras).

- [ ] ✅ DL Spec Course 1 COMPLETE
- [ ] ✅ Churn prediction project with advanced evaluation — pushed to GitHub

### Day 42 (Sunday — 6-Week Milestone) ⏰ 2hrs
**Milestone Review + Thread**

Six weeks in. Assess your progress:
- [ ] Ng ML Specialization complete (3 courses) ✅
- [ ] Karpathy micrograd + all 5 makemore parts ✅
- [ ] Ng DL Spec Course 1 complete ✅
- [ ] 5 Kaggle certificates ✅
- [ ] 3 Kaggle competition/project submissions ✅
- [ ] 20+ LeetCode problems ✅
- [ ] DSA: sorting, graph search, shortest paths ✅

Write a milestone X thread. Update your GitHub portfolio comprehensively.

---

# ═══════════════════════════════════════════
# PHASE 3B: DEEP LEARNING MASTERY
# Weeks 7–8 (Days 43–56)
# ═══════════════════════════════════════════

## WEEK 7: Build GPT + Ng DL Spec Courses 2-3

| Day | Focus | Hours | Deliverable |
|-----|-------|-------|------------|
| 43 Mon | Ng DL Spec Course 2: Improving Deep NNs — Weeks 1-2 (optimization algorithms: momentum, RMSprop, Adam; regularization: dropout, L2, early stopping) | 3 | Course 2 Weeks 1-2 complete |
| 44 Tue | Ng DL Spec Course 2: Weeks 3-4 (hyperparameter tuning, batch normalization, multi-class with softmax, TensorFlow). Complete Course 2. | 3 | Course 2 COMPLETE |
| 45 Wed | **KARPATHY: "Let's build GPT: from scratch, in code, spelled out"** — First half. Code along: bigram model → self-attention mechanism → single head of attention → multi-head attention. This lecture is ~2 hours. | 4 | Self-attention + multi-head attention implemented |
| 46 Thu | **Karpathy GPT continued** — Second half: feed-forward layers, residual connections, layer normalization, full transformer block, training loop. By end of day you have a working GPT generating text. | 4 | **Complete GPT model generating text** |
| 47 Fri | Rebuild GPT from memory. Start with empty file. Implement: token embeddings, positional embeddings, self-attention, multi-head attention, feed-forward, transformer block, full GPT. Test: generate text. Where you get stuck = study more. | 4 | GPT rebuilt without tutorial |
| 48 Sat | Ng DL Spec Course 3: "Structuring ML Projects" (complete — it's short). Then start Course 4: "CNNs" Weeks 1-2 (convolution operation, padding, stride, pooling, classic architectures: LeNet, AlexNet, VGG). | 5 | Course 3 COMPLETE + Course 4 Weeks 1-2 |
| 49 Sun | Review GPT implementation. Write detailed README explaining every component. Update GitHub. LeetCode: 2 problems. | 2 | GPT README + LeetCode |

## WEEK 8: Complete DL Spec + CNN/RNN Projects

| Day | Focus | Hours | Deliverable |
|-----|-------|-------|------------|
| 50 Mon | Ng DL Spec Course 4: Weeks 3-4 (ResNets, Inception, transfer learning, object detection — YOLO, face recognition, neural style transfer). Complete Course 4. | 3 | Course 4 COMPLETE |
| 51 Tue | Ng DL Spec Course 5: "Sequence Models" Weeks 1-2 (RNNs, GRUs, LSTMs, bidirectional RNNs, deep RNNs, word embeddings, Word2Vec, GloVe). | 3 | Course 5 Weeks 1-2 |
| 52 Wed | Ng DL Spec Course 5: Weeks 3-4 (Attention mechanism, Transformer architecture, positional encoding). This is Ng's version of what Karpathy taught hands-on — now you get the formal mathematical treatment. Complete Course 5. | 3 | **Ng DL Specialization — ALL 5 COURSES COMPLETE** ✅ |
| 53 Thu | PROJECT: Train a CNN image classifier on CIFAR-10 using PyTorch from scratch (not using pre-built models). Implement: Conv2d layers, pooling, fully connected layers, training loop, validation, data augmentation. Target: >85% test accuracy. | 4 | CNN classifier — pushed to GitHub |
| 54 Fri | PROJECT: Implement a simple RNN/LSTM for text generation in PyTorch. Train on a text corpus of your choice. Compare the output quality to your Karpathy GPT model. | 4 | RNN text generator — pushed to GitHub |
| 55 Sat | PROJECT: Transfer learning — take a pre-trained ResNet, fine-tune it on a small custom dataset (use any image classification dataset from Kaggle with <10 classes). This shows the power of not training from scratch. | 4 | Transfer learning notebook — pushed to GitHub |
| 56 Sun | **8-WEEK MILESTONE.** Both Ng specializations complete. Karpathy complete. GPT built from scratch. Write a comprehensive milestone post. | 2 | Milestone thread + portfolio update |

---

# ═══════════════════════════════════════════
# PHASE 4: MODERN AI — TRANSFORMERS & LLMs
# Weeks 9–12 (Days 57–84)
# ═══════════════════════════════════════════

## WEEK 9: MIT 6.S191 + CS231n

| Day | Focus | Hours | Deliverable |
|-----|-------|-------|------------|
| 57 Mon | MIT 6.S191 Lectures 1-2 (Intro to Deep Learning + Sequence Modeling). Complete Lab 1 on Google Colab. The MIT labs are self-contained and well-structured — run every cell. | 3 | Lab 1 complete |
| 58 Tue | MIT 6.S191 Lectures 3-4 (CNNs + Deep Generative Modeling — VAEs, GANs, Diffusion). Complete Lab 2. | 3 | Lab 2 complete |
| 59 Wed | MIT 6.S191 Lectures 5-6 (Reinforcement Learning + LLMs & New Frontiers). Complete remaining labs. This gives you a unified view of the entire modern DL landscape. | 3 | All MIT labs complete |
| 60 Thu | CS231n Assignment 1: Start. Implement k-Nearest Neighbors, SVM linear classifier, Softmax classifier, Two-layer neural network — all from scratch. The CS231n assignments are legendary — they force deep understanding. | 4 | A1 partially complete |
| 61 Fri | CS231n Assignment 1: Complete. Debug your implementations. Run all sanity checks. | 4 | A1 fully complete — pushed to GitHub |
| 62 Sat | PROJECT: Fine-tune nanoGPT (Karpathy's lightweight GPT) on a domain-specific corpus. Choose a corpus related to legal text, Nigerian law, or AI safety. Generate text and evaluate quality. This connects your ML skills to your domain. | 5 | Fine-tuned nanoGPT on custom corpus |
| 63 Sun | Review week. Continue Stanford Algorithms (if behind). LeetCode: 2 problems. | 2 | Week review + LeetCode |

## WEEK 10: Transformers Deep Dive

| Day | Focus | Hours | Deliverable |
|-----|-------|-------|------------|
| 64 Mon | Read "Attention Is All You Need" (Vaswani et al., 2017). Read the Harvard NLP annotated transformer (nlp.seas.harvard.edu/annotated-transformer/). Take detailed notes. | 3 | Written paper summary with key equations |
| 65 Tue | Implement self-attention from scratch in PyTorch. Not using nn.MultiheadAttention — build the Q, K, V projections, scaled dot-product attention, and masking manually. | 3 | Self-attention from scratch notebook |
| 66 Wed | Implement a complete transformer encoder block from scratch: multi-head attention + feed-forward + residual connections + layer normalization. Stack 4 blocks. Test on a simple classification task. | 4 | Transformer encoder from scratch |
| 67 Thu | Hugging Face NLP Course Chapters 1-3: The Transformer pipeline, using pre-trained models, fine-tuning with the Trainer API. This shows you the production side — how to use transformers efficiently without building from scratch. | 3 | HF pipeline working on a real task |
| 68 Fri | HF NLP Course Chapters 4-5: Tokenizers (BPE, WordPiece, SentencePiece) and the Datasets library. Understand how text becomes numbers. Build a custom tokenizer. | 3 | Custom tokenizer + dataset processing |
| 69 Sat | PROJECT: Fine-tune BERT (or DistilBERT for speed) for text classification on a legal dataset. Use a legal text classification dataset from Hugging Face Hub or create one from public court documents. Evaluate with precision/recall/F1. | 5 | Fine-tuned BERT on legal text — pushed to GitHub |
| 70 Sun | Write a blog post or detailed README: "Transformers from Scratch to Fine-Tuning." Cover self-attention math, implementation, and practical fine-tuning. | 2 | Published post/README |

## WEEK 11: LLMs, Fine-Tuning & RAG

| Day | Focus | Hours | Deliverable |
|-----|-------|-------|------------|
| 71 Mon | HF NLP Course Chapters 6-8: complete the course. Focus on the chapter about training from scratch and the real-world deployment considerations. | 3 | HF NLP Course COMPLETE |
| 72 Tue | DeepLearning.AI short course: "Fine-Tuning Large Language Models" (free on deeplearning.ai). Learn: when to fine-tune vs. prompt, data preparation for fine-tuning, training configuration, evaluation. | 3 | Course complete with notes |
| 73 Wed | Learn QLoRA and PEFT: Read the Hugging Face PEFT documentation. Understand: LoRA (low-rank adaptation), QLoRA (quantized LoRA), why these enable fine-tuning large models on consumer GPUs. Implement: fine-tune Qwen 0.5B with QLoRA on Google Colab. | 4 | QLoRA fine-tuning notebook |
| 74 Thu | Fine-tune a slightly larger model (Mistral 7B or Llama 3 8B) with QLoRA on a small legal Q&A dataset. Use 4-bit quantization + LoRA with rank 16. This should run on Colab Pro's A100 or on a free T4 with small batch sizes. | 4 | Fine-tuned 7B+ model |
| 75 Fri | DeepLearning.AI: "LangChain for LLM Application Development" (free). Learn RAG (Retrieval-Augmented Generation) concepts: document loading, text splitting, embedding, vector stores, retrieval chain. | 3 | RAG concepts understood + prototype |
| 76 Sat | PROJECT: Build a complete RAG system for legal document Q&A. Use LlamaIndex or LangChain. Load real legal documents (e.g., Nigerian law PDFs or public court decisions), chunk them, embed them, and build a retrieval pipeline that grounds LLM answers in the source documents. | 5 | Working legal RAG system — pushed to GitHub |
| 77 Sun | Review. This is a major portfolio piece — polish the RAG project, write comprehensive documentation. | 2 | RAG project documented and polished |

## WEEK 12: Reinforcement Learning + Alignment Foundations

| Day | Focus | Hours | Deliverable |
|-----|-------|-------|------------|
| 78 Mon | HF Deep RL Course: Units 1-2. RL framework (states, actions, rewards, policies), Q-Learning, deep Q-networks. Complete the hands-on exercises — train agents in simple environments. | 3 | RL exercises complete |
| 79 Tue | HF Deep RL Course: Units 3-4. Deep Q-Learning, Policy Gradient methods (REINFORCE). These are the building blocks for RLHF. | 3 | Policy gradient agent trained |
| 80 Wed | HF Deep RL Course: Unit 5 + PPO. Proximal Policy Optimization is the algorithm behind RLHF in ChatGPT. Understand: why PPO clips the policy gradient, the role of the KL penalty, the reward model. | 3 | PPO implementation understood |
| 81 Thu | HF Deep RL Course: RLHF bonus unit. Then read: "Training language models to follow instructions with human feedback" (InstructGPT paper, Ouyang et al.). Take detailed notes on the 3-step pipeline: SFT → Reward Model → PPO. | 3 | InstructGPT paper notes + RLHF understanding |
| 82 Fri | Read: "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., Anthropic) + "Direct Preference Optimization" (Rafailov et al.). Take detailed notes. Map out the DPO loss function mathematically. | 3 | CAI + DPO paper notes |
| 83 Sat | PROJECT: Implement DPO training on a small model. Use Hugging Face TRL's DPOTrainer. Create a small preference dataset (or use UltraFeedback). Fine-tune Qwen 0.5B or similar. Compare outputs before and after DPO. | 5 | DPO implementation — pushed to GitHub |
| 84 Sun | **12-WEEK MILESTONE.** Write your most comprehensive thread yet. You've now completed: 2 Ng specializations, Karpathy series, MIT 6.S191, HF NLP/RL courses, built GPT from scratch, fine-tuned LLMs, built RAG, implemented DPO. | 2 | Milestone post + full portfolio update |

---

# ═══════════════════════════════════════════
# PHASE 5: ALIGNMENT & CONSTITUTIONAL AI
# Weeks 13–18 (Days 85–126)
# ═══════════════════════════════════════════

## WEEKS 13–14: Constitutional AI Pipeline (Days 85–98)

This is where your unique legal AI expertise meets your newly-built ML skills. You're building something that doesn't exist yet: a constitutionally-aligned legal AI system.

| Day | Focus | Hours | Deliverable |
|-----|-------|-------|------------|
| 85 | Write your legal AI constitution. Draft 15-25 principles covering: citation accuracy, jurisdictional awareness, epistemic honesty, appropriate uncertainty, avoiding hallucinated case law. Each principle should be specific enough that a model could evaluate against it. Study Anthropic's published constitutional principles for inspiration but make yours domain-specific. | 3 | Constitution v1 document |
| 86 | Refine the constitution. For each principle, write 2-3 example violations and 2-3 examples of compliance. This becomes your evaluation rubric. | 3 | Constitution v2 with examples |
| 87 | Generate seed prompts: write 100+ legal questions spanning different areas (contract law, constitutional law, criminal procedure, etc.) and difficulty levels. These are the prompts your model will respond to during the critique-revision loop. | 3 | Seed prompt dataset |
| 88 | Generate initial model responses: run your seed prompts through a base model (use a small open-weight model or API). Collect all responses. | 3 | Initial response dataset |
| 89 | Build the critique loop: for each response, use a stronger model (Claude or GPT-4 via API) to evaluate against each constitutional principle. Generate revised responses that better adhere to the constitution. | 4 | Critique-revision pipeline working |
| 90 | Generate preference pairs from the critique-revision loop. Original responses = "rejected," revised responses = "chosen." Clean the data into the format TRL's DPOTrainer expects. | 3 | Preference dataset ready |
| 91 | Train DPO: fine-tune your base model on the preference pairs using QLoRA + DPO. Monitor training loss. Save checkpoints. | 4 | Aligned model v1 |
| 92 | Evaluate: benchmark v1 against the base model. Metrics: rate of hallucinated citations, jurisdictional accuracy, appropriate uncertainty expression. Use your constitution's evaluation rubric. | 3 | Evaluation results documented |
| 93 | Iterate: analyze failure cases from evaluation. Refine constitutional principles that aren't working. Generate additional preference pairs targeting weak areas. | 3 | Refined constitution + additional data |
| 94 | Retrain: DPO round 2 with expanded dataset. Compare v2 to v1 and base model. | 4 | Aligned model v2 with comparison |
| 95-96 | Add RAG grounding: integrate your aligned model with the legal RAG system from Week 11. The constitution should penalize responses not grounded in retrieved documents. Test the full pipeline. | 5+5 | RAG-grounded constitutional AI system |
| 97-98 | Write up results. Structure as: Problem → Approach → Constitution Design → Pipeline → Results → Limitations. Target Alignment Forum quality. | 4+4 | Draft blog post / Alignment Forum post |

## WEEKS 15–16: Mechanistic Interpretability (Days 99–112)

| Day | Focus | Hours | Deliverable |
|-----|-------|-------|------------|
| 99-100 | ARENA self-study Chapter 1: Transformer Circuits. Install TransformerLens (Neel Nanda's library). Work through the introductory tutorials — loading models, extracting activations, visualizing attention patterns. | 3+3 | TransformerLens setup + basic analysis |
| 101-102 | Replicate the induction head finding exercise from ARENA. Induction heads are attention heads that implement a simple "A...B...A→B" pattern — they're one of the clearest known circuits in transformers. Finding them teaches you how to identify circuits. | 3+3 | Induction heads identified and visualized |
| 103-104 | Apply interpretability to YOUR domain: analyze attention patterns when a language model processes legal reasoning prompts. Where does the model attend when generating citations? When generating jurisdictional qualifiers? | 4+4 | Legal reasoning interpretability notebook |
| 105-106 | Read Anthropic's "Scaling Monosemanticity" (sparse autoencoders paper). Take detailed notes. Understand: superposition, sparse features, how autoencoders can extract interpretable features from model activations. | 3+3 | Detailed paper notes |
| 107-108 | Implement a basic sparse autoencoder on a small model's activations. This is cutting-edge research territory — your implementation doesn't need to replicate Anthropic's scale, but it should demonstrate the concept. | 4+4 | Sparse autoencoder implementation |
| 109-110 | Write up your interpretability findings. If you found anything interesting about legal reasoning circuits, this could be a genuine contribution to the field. | 3+3 | Write-up / blog post |
| 111-112 | Apply to SPAR (sparai.org) and any other open programs. Your portfolio now includes: constitutional AI for legal accuracy, interpretability analysis, DPO implementation — this is a strong application. | 3+3 | Applications submitted |

## WEEKS 17–18: Strategy + Capstone Preparation (Days 113–126)

| Day | Focus | Hours | Deliverable |
|-----|-------|-------|------------|
| 113-119 | BlueDot AGI Strategy or Technical AI Safety Course (if accepted — these run as cohorts with weekly discussions). Continue your daily study alongside the cohort. If not yet accepted, self-study the BlueDot curriculum at bluedot.org (it's partially open). Also: Stanford Algorithms Specialization — complete remaining courses (you should be on Course 3 or 4 by now). LeetCode: maintain 3-5 problems/week. | ~3/day | Course participation + algorithms progress |
| 120-122 | Begin capstone project planning: "Constitutional AI for Legal Accuracy" — the polished, publishable version. Define scope, gather a proper evaluation dataset, design rigorous experiments. This isn't a new project — it's the production-quality version of what you built in Weeks 13-14. | 4+4+4 | Capstone project plan + evaluation framework |
| 123-124 | Build a proper evaluation benchmark for legal AI hallucination. Create test cases with known-correct answers. Include: factual accuracy, citation validity, jurisdictional appropriateness, uncertainty calibration. | 4+4 | Evaluation benchmark |
| 125-126 | Capstone v1 pipeline: end-to-end system working — seed prompts → constitutional critique → preference pairs → DPO training → RAG-grounded inference → evaluation. Automated as much as possible. | 5+3 | Capstone pipeline v1 running |

---

# ═══════════════════════════════════════════
# PHASE 6: CAPSTONE + PORTFOLIO + LAUNCH
# Weeks 19–24 (Days 127–168)
# ═══════════════════════════════════════════

## WEEKS 19–21: Capstone Execution (Days 127–147)

Dedicate 4-5 hours daily to the capstone project. This is your flagship portfolio piece — the project that demonstrates everything you've learned and positions you uniquely at the intersection of law and AI alignment.

| Week | Focus | Key Outputs |
|------|-------|------------|
| 19 | Run full experiments. Train multiple versions with different constitutions. Ablation studies: which principles help most? How many preference pairs are needed? What base model performs best? | Experimental results + analysis |
| 20 | Polish the system. Handle edge cases. Improve evaluation. Run final benchmarks. Create a demo (Gradio or Streamlit app showing before/after alignment). | Demo app + final benchmarks |
| 21 | Write the paper/post. Structure: Abstract → Introduction → Related Work → Method → Experiments → Results → Discussion → Limitations → Conclusion. Target: Alignment Forum or a workshop paper. | Published Alignment Forum post or preprint |

## WEEKS 22–24: Portfolio Polish + Applications (Days 148–168)

| Week | Focus | Key Outputs |
|------|-------|------------|
| 22 | Polish GitHub portfolio. Every major project should have a clear README with: problem statement, approach, results, how to reproduce. Create a personal website or portfolio page linking everything. Record a 5-minute demo video of your capstone. | Polished portfolio + demo video |
| 23 | Applications: ARENA 7.0 (Jan 2026 London — application deadline likely Oct 2025), MATS (next open cohort), SPAR (if not already in), research positions at alignment organizations. Your portfolio now includes: GPT from scratch, fine-tuned LLMs, RAG system, constitutional AI pipeline, interpretability analysis. | Applications submitted |
| 24 | Continue DSA (target 150+ LeetCode problems total). Start engaging deeply with Alignment Forum — write comments on recent posts, share your work. Network with alignment researchers on X. Begin thinking about your next research question. | Community engagement + ongoing learning |

---

# ═══════════════════════════════════════════
# WEEKS 25–30: ADVANCED + CONTINUOUS GROWTH
# ═══════════════════════════════════════════

## Ongoing Weekly Structure (After Week 24)

| Activity | Frequency | Purpose |
|----------|-----------|---------|
| Read 1 alignment/ML paper | Weekly | Stay current |
| LeetCode: 3-5 problems | Weekly | DSA maintenance |
| Alignment Forum: write or comment | Bi-weekly | Community presence |
| New mini-project or experiment | Bi-weekly | Continuous skill building |
| Review and refine capstone | Monthly | Iterative improvement |
| Apply to programs and positions | Ongoing | Career advancement |

## Target Projects for Weeks 25–30:
1. Extend capstone with new legal domains or jurisdictions
2. Build a multi-agent debate system for legal reasoning verification
3. Contribute to an open-source alignment tool (TRL, TransformerLens, etc.)
4. Write a policy paper on "AI Alignment Lessons from Constitutional Law"
5. Prepare for ARENA 7.0 (if accepted) — review curriculum materials

---

# ═══════════════════════════════════════════
# PROGRESS TRACKER
# ═══════════════════════════════════════════

Check each box as you complete it. Review this table every Sunday.

| Week | Phase | Key Milestone | Done |
|------|-------|--------------|------|
| 1 | Foundations | Linear algebra + Python + Pandas + first EDA project | ⬜ |
| 2 | Foundations | Calculus + Probability + Loss functions + 5 Kaggle certs | ⬜ |
| 3 | Foundations | DSA started + Titanic Kaggle competition submitted | ⬜ |
| 4 | ML | Ng ML Spec Course 1 + Linear/Logistic regression from scratch + House Prices | ⬜ |
| 5 | ML/DL | Ng Course 2 + **Micrograd from scratch** + makemore Part 1 | ⬜ |
| 6 | ML/DL | Karpathy makemore complete (all 5 parts) + Ng ML Spec DONE + Churn project | ⬜ |
| 7 | DL | **GPT built from scratch** + Ng DL Spec Courses 2-3 | ⬜ |
| 8 | DL | Ng DL Spec COMPLETE (all 5 courses) + CNN + RNN + Transfer Learning projects | ⬜ |
| 9 | Advanced | MIT 6.S191 all labs + CS231n Assignment 1 + Fine-tuned nanoGPT | ⬜ |
| 10 | Transformers | Transformer from scratch + HF NLP Course + Fine-tuned BERT on legal data | ⬜ |
| 11 | LLMs | QLoRA fine-tuning + RAG system for legal Q&A | ⬜ |
| 12 | RL/Alignment | HF RL Course + DPO implementation + InstructGPT/CAI papers | ⬜ |
| 13-14 | Alignment | **Constitutional AI pipeline for legal accuracy** complete | ⬜ |
| 15-16 | Interpretability | Mechanistic interp + sparse autoencoders + SPAR application | ⬜ |
| 17-18 | Strategy | BlueDot course + capstone planning + evaluation benchmark | ⬜ |
| 19-21 | Capstone | **Capstone project complete + Alignment Forum post published** | ⬜ |
| 22-24 | Portfolio | GitHub polished + demo video + ARENA/MATS applications submitted | ⬜ |
| 25-30 | Advanced | Ongoing research + community engagement + program participation | ⬜ |

---

# ═══════════════════════════════════════════
# WHEN MOTIVATION DROPS (AND IT WILL)
# ═══════════════════════════════════════════

Read this section when you hit a wall. Bookmark it.

**Week 2-3 wall:** "This math is boring, I want to build things." Push through. The math is a 3-week investment that prevents 6 months of confusion. You're 90% done with the boring part.

**Week 5-6 wall:** "Backpropagation is too hard, I don't get it." You're supposed to not get it yet. Karpathy's micrograd lecture + rebuilding from memory is what makes it click. Trust the process.

**Week 8-10 wall:** "There's too much to learn, I'll never catch up." You've already completed 2 full specializations and built GPT from scratch. You're further than 95% of people who start learning ML. The field is vast — nobody knows everything.

**Week 13-16 wall:** "My constitutional AI results aren't impressive." First results are never impressive. The point is you built the pipeline. Iterate. The second version is always dramatically better than the first.

**Any day wall:** "I missed yesterday and today I don't feel like it." This is the critical moment. Do the absolute minimum: open one notebook, write one line of code, solve one LeetCode easy. Break the chain of inaction. Tomorrow will be easier.

**The truth:** Every ML researcher and engineer has felt exactly what you're feeling. The difference between people who make it and people who don't is not talent — it's consistency through the hard weeks. You have 30 weeks mapped out. All you need to do is show up each day and do the next task on the list.

---

**Start date: _____________**
**30-week target completion: _____________**
**Daily X hashtag: #BukunmiBuildsAI**
**GitHub repo: github.com/[username]/ml-journey**
