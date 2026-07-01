import math
import numpy as np
import scipy
import torch
from sentence_transformers import SentenceTransformer

# example document
documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.'
]

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(documents)

print(embeddings.shape)

def euclidean_distance_fn(vector1, vector2):
    squared_sum = sum((x - y) ** 2 for x, y in zip(vector1, vector2))
    return math.sqrt(squared_sum)

# distance b/w two array it doesnot matter what vector we put first
print(euclidean_distance_fn(embeddings[0], embeddings[1]))
print(euclidean_distance_fn(embeddings[1], embeddings[0]))

# distance between all the vectors
l2_dist_manual = np.zeros([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        l2_dist_manual[i,j] = euclidean_distance_fn(embeddings[i],embeddings[j])

print(l2_dist_manual)

# upper code is not efficent, as the i and j which ever we put first the distance will be same
# so we will calculate it once
l2_dist_manual_imp = np.zeros([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        if j > i:
            l2_dist_manual_imp[i,j] = euclidean_distance_fn(embeddings[i],embeddings[j])
        elif i > j: # copy the upper triange to the lower triangle
            l2_dist_manual_imp[i,j] = l2_dist_manual[j,i]

print(l2_dist_manual_imp)

# l2 dist using scipy
l2_dist_scipy = scipy.spatial.distance.cdist(embeddings,embeddings,'euclidean')
print(l2_dist_scipy)

# verifying that l2_dist_manual and 12_dist_spatical are identical
np.allclose(l2_dist_manual,l2_dist_scipy)


# DOT PRODUCT SIMILARITY AND DISTANCE
def dot_product_fn(vector1, vector2):
    return sum(x * y for x, y in zip(vector1, vector2))

dot_product_manual = np.empty([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        dot_product_manual[i,j] = dot_product_fn(embeddings[i], embeddings[j])

print(dot_product_manual)

# Calucltaing the dot product wung matrix multiplication
dot_product_operator = embeddings @ embeddings.T
print(dot_product_operator)

# Verifying equal
np.allclose(dot_product_manual, dot_product_operator, atol=1e-05)

# Equivalent to `np.matmul()` if both arrays are 2-D:
np.matmul(embeddings,embeddings.T)

# calculating dot product distance
dot_product_distance = -dot_product_manual
print(dot_product_distance)


# Cosine similrity and search

# Calculating L2 norm
l2_norms = np.sqrt(np.sum(embeddings**2, axis=1))

# l2 norm reshaped
l2_norms_reshaped = l2_norms.reshape(-1,1)

# Normalizing
normalized_embeddings_manual = embeddings/l2_norms_reshaped
np.sqrt(np.sum(normalized_embeddings_manual**2, axis=1))

# Normalized embeddings using PyTorch
normalized_embeddings_torch = torch.nn.functional.normalize(
    torch.from_numpy(embeddings)
).numpy()
print(normalized_embeddings_torch)

print(np.allclose(normalized_embeddings_manual, normalized_embeddings_torch))

# Caluculating cosine similarity manual
cosine_similarity_manual = np.empty([4,4])
for i in range(normalized_embeddings_manual.shape[0]):
    for j in range(normalized_embeddings_manual.shape[0]):
        cosine_similarity_manual[i,j] = dot_product_fn(
            normalized_embeddings_manual[i], 
            normalized_embeddings_manual[j]
        )

print(cosine_similarity_manual)

# Calculating cosine similarity using matrix multiplication
cosine_similarity_operator = normalized_embeddings_manual @ normalized_embeddings_manual.T

print(np.allclose(cosine_similarity_manual, cosine_similarity_operator))

# Calculating cosine disance
print(1 - cosine_similarity_manual)


# SIMILARITY SEARCH using a query
query_embedding = model.encode(
    ["Who is responsible for a coding project and fixing others' mistakes?"]
)

# Second, normalize the query embedding:
normalized_query_embedding = torch.nn.functional.normalize(
    torch.from_numpy(query_embedding)
).numpy()

# Third, calculate the cosine similarity between the documents and the query by using the dot product:
cosine_similarity_q3 = normalized_embeddings_manual @ normalized_query_embedding.T

# Fourth, find the position of the vector with the highest cosine similarity:
highest_cossim_position = cosine_similarity_q3.argmax()

# Fifth, find the document in that position in the `documents` array:
documents[highest_cossim_position]


