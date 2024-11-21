import torch
from transformers import LlamaForCausalLM
import numpy as np
from sklearn.decomposition import PCA

llm = LlamaForCausalLM.from_pretrained('path/to/llm')

llama_embeds = llm.get_input_embeddings().weight.data

numpy_matrix = llama_embeds.numpy()

pca = PCA(n_components=1000)
pca.fit(numpy_matrix)

explained_variance_ratio = pca.explained_variance_ratio_

ratio_sum = 0
for i, ratio in enumerate(explained_variance_ratio):
    ratio_sum += ratio
print(ratio_sum)

components = pca.components_

components_float16 = components.astype(np.float16)

tensor_components_float16 = torch.tensor(components_float16)
torch.save(tensor_components_float16, './PCA_1000_pc_llama.pt')
