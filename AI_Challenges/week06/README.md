#  Week 6 Challenge - Calculate the Bert model parameters.
## Composition of Bert model
1. Embedding layer
- Token embeddings
- Segment embeddings
- Position embeddings
2. Encoder
3. Pooling layer
## Main steps of parameter calculation
1. Calculate the parameters of the embedding layer
- Calculate the parameters of the word embedding layer
- Assuming the vocabulary size is V, the embedding dimension is d_{\text{model}}, the parameter of the word embedding is: V \times d_{\text{model}}
- Calculate the parameters of the segment embedding layer,
- Assuming the maximum sequence length is L, the embedding dimension is d_{\text{model}}, the parameter of the position embedding is: L \times d_{\text{model}}
- Calculate the parameters of the position embedding layer
- Segment embedding usually has two types (sentence A and sentence B), the embedding dimension is d_{\text{model}}, and the number of parameters is: 2 \times d_{\text{model}}
- The number of parameters of the total embedding layer
-\text{Embedding layer parameters} = V \times d_{\text{model}} + L \times d_{\text{model}} + 2 \times d_{\text{model}}

2. Calculate the number of parameters of the encoder
- Multi-Head Attention
- Each attention head has three weight matrices: query (Q), key (K) and value (V), as well as the output weight matrix.
- Assuming the number of attention heads is h, and the dimension of each head is d_k = d_{\text{model}} / h, then:
- The parameters of Q, K, and V are: d_{\text{model}} \times d_k \quad (\text{each head}) \quad \rightarrow \quad h \cdot d_{\text{model}} \cdot d_k
- The output weight matrix of multi-head attention is: d_{\text{model}} \times d_{\text{model}}
- The total number of multi-head attention parameters is: 3 \cdot h \cdot d_{\text{model}} \cdot d_k + d_{\text{model}} \cdot d_{\text{model}}

- Feed-Forward Neural Network
- The forward network consists of two fully connected layers, the first layer will d_{\text{model}} maps to d_{\text{ffn}} , and the second layer maps d_{\text{ffn}} back to d_{\text{model}} : d_{\text{model}} \times d_{\text{ffn}} + d_{\text{ffn}} \times d_{\text{model}}
- Total forward network parameters: 2 \cdot d_{\text{model}} \cdot d_{\text{ffn}}
- Layer Normalization:
- Each layer has two normalization modules, each of which requires two parameters (scaling and offset), and the total number of parameters is: 2 \cdot d_{\text{model}} \quad (\text{per layer})
- Encoder parameters
- \text{Encoder parameters} = 3 \cdot h \cdot d_{\text{model}} \cdot d_k + d_{\text{model}} \cdot d_{\text{model}} + 2 \cdot d_{\text{model}} \cdot d_{\text{ffn}} + 2 \cdot d_{\text{model}} \quad (\text{per layer})
3. Calculate the number of parameters of the pooling layer
- The pooling layer is a fully connected layer that transforms the output of [CLS] into a fixed-size sentence vector. The number of parameters is: d_{\text{model}} \times d_{\text{model}}

Example: Bert-base
Parameter settings:
- V = 30,000
- L = 512
- d_{\text{model}} = 768
- h = 12
- d_k = d_{\text{model}} / h = 64
- d_{\text{ffn}} = 3072
- N = 12

1. Embedding layer parameters
\text{Embedding layer parameters} = 30,000 \cdot 768 + 512 \cdot 768 + 2 \cdot 768 = 23,685,120
2. Encoder parameters (single layer)
\text{Single layer encoder parameters} = 3 \cdot 12 \cdot 768 \cdot 64 + 768^2 + 2 \cdot 768 \cdot 3072 + 2 \cdot 768
= 1,769,472 + 589,824 + 4,718,592 + 1,536 = 7,079,424
3. Total encoder parameters
\text{Total encoder parameters} = 12 \cdot 7,079,424 = 84,952,608
4. Pooling layer parameters
\text{Pooling layer parameters} = 768^2 = 589,824
5. Total parameters
\text{Total parameters} = 23,685,120 + 84,952,608 + 589,824 = 109,227,552

# Summary: For any BERT model, the total number of parameters is:
\text{Total parameters} = V \cdot d_{\text{model}} + L \cdot d_{\text{model}} + 2 \cdot d_{\text{model}} + N \cdot \left( 3 \cdot h \cdot d_{\text{model}} \cdot \frac{d_{\text{model}}}{h} + d_{\text{model}}^2 + 2 \cdot d_{\text{model}} \cdot d_{\text{ffn}} + 2 \cdot d_{\text{model}} \right) + d_{\text{model}}^2