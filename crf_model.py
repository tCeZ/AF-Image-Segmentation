import numpy as np
import pycrfsuite

def create_unary_features(image):
    # Extract simple unary features such as intensity and gradient
    intensity = image.flatten()
    gradient_x = np.gradient(image, axis=0).flatten()
    gradient_y = np.gradient(image, axis=1).flatten()
    
    features = []
    for i, val in enumerate(intensity):
        features.append({
            f'intensity_{val}': 1,
            f'gradient_x_{gradient_x[i]}': 1,
            f'gradient_y_{gradient_y[i]}': 1,
        })
    return features

def create_pairwise_features(image):
    # Example pairwise feature: horizontal and vertical pairwise connections
    features = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighbors = []
            if i > 0:
                neighbors.append((i - 1, j))
            if i < image.shape[0] - 1:
                neighbors.append((i + 1, j))
            if j > 0:
                neighbors.append((i, j - 1))
            if j < image.shape[1] - 1:
                neighbors.append((i, j + 1))
            features.append(neighbors)
    return features

def prepare_data(image, ground_truth):
    X = [create_unary_features(image)]
    y = [ground_truth.flatten().tolist()]
    edges = create_pairwise_features(image)
    return X, y, edges

def train_crf(X, y, edges):
    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq, edge in zip(X, y, edges):
        trainer.set_params({
            'c1': 0.1,  # coefficient for L1 penalty
            'c2': 0.1,  # coefficient for L2 penalty
            'num_memories': 6,  # number of truncated Newton steps in L-BFGS update
            'max_iterations': 200,  # stop earlier
            'feature.possible_transitions': True,
            'feature.possible_states': True
        })

        trainer.append(xseq, yseq, edge)

    trainer.train('crf_model.crfsuite')

def predict_crf(image):
    tagger = pycrfsuite.Tagger()
    tagger.open('crf_model.crfsuite')

    unary_features = create_unary_features(image)
    y_pred = tagger.tag(unary_features)

    return np.array(y_pred).reshape(image.shape)

# Create toy images and ground truth masks
image_train = np.random.randint(0, 256, (10, 10))
ground_truth_train = (image_train > 128).astype(int)

image_test = np.random.randint(0, 256, (10, 10))

# Prepare data
X, y, edges = prepare_data(image_train, ground_truth_train)

# Train the CRF model
train_crf(X, y, edges)

# Perform prediction on a new image
predicted_mask = predict_crf(image_test)

print(predicted_mask)

