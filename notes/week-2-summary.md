# Week 2: Calculus, Probability & Statistics

## Calculus
- Derivatives: numerical derivative function from scratch — approximated derivatives for x², sin(x), e^x and compared to analytical solutions
- Chain rule: df/dx = df/dg * dg/dx — verified numerically that the product of local derivatives equals the full derivative
- Gradient descent from scratch on f(x) = (x-3)² + 2 with path visualization
- Key insight: the chain rule is the backbone of backpropagation — it lets you compute dLoss/dWeight by multiplying local derivatives along the path from weight to loss, without ever expanding the full network into one massive expression

## Probability & Statistics
- Bayes' theorem implemented as a function — solved the classic disease testing problem
- Distribution simulations: biased coin flips converging to expected probability over 10,000 trials
- Normal distributions plotted with varying mean and standard deviation

## Information Theory & Loss Functions
- Binary cross-entropy from scratch: BCE = -[y*log(p) + (1-y)*log(1-p)]
- Categorical cross-entropy from scratch (multi-class)
- Mean squared error from scratch (regression)
- Visualized how BCE loss explodes toward infinity as predictions become confidently wrong
- Key insight: BCE has two halves — y acts as a switch. When y=1, it punishes false negatives. When y=0, it punishes false positives. The punishment is logarithmic, not linear — confidently wrong predictions are catastrophically penalized

## Kaggle Certificates
- Intro to Machine Learning — decision trees, random forests, model validation, underfitting/overfitting
- Intermediate Machine Learning — missing values (3 strategies), categorical encoding (ordinal vs one-hot), pipelines, cross-validation, XGBoost
- Extended notebooks: hyperparameter exploration for max_leaf_nodes, XGBoost vs random forest comparison

## LeetCode (DSA Start)
- 2 problems solved
- **Two Sum (#1)** — hash map (dict) pattern: store seen values as keys with their indices as values, check for complement at each step. O(n) time / O(n) space. Key trade-off: spent extra memory to eliminate the O(n²) brute-force pair checking
- **Best Time to Buy and Sell Stock (#121)** — tracking pattern: maintain running minimum price, compute profit at each step, track max profit. O(n) time / O(1) space. Key lesson: not every problem needs a hash map — this is a tracking problem, not a lookup problem

## Key Concepts Solidified
- **Chain rule:** dh/dx = df/dg * dg/dx — "with respect to" matters; each piece is a rate of change of one function with respect to its own input
- **Activation functions:** without nonlinearities (ReLU, sigmoid), stacking linear layers collapses to a single linear layer — depth becomes meaningless
- **Underfitting vs overfitting:** too simple (can't capture signal) vs too complex (memorizes noise). Diagnosed by comparing train vs validation performance
- **Ordinal vs one-hot encoding:** ordinal implies ranking and distance between categories — only use when a natural order exists. One-hot treats each category independently
- **Time-space trade-off:** Two Sum trades O(n) space for O(n) time. Buy/Sell Stock needs neither — just two variables
