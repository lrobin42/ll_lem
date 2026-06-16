use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Int, Tensor, backend::Backend};
use burn::tensor::{Shape, TensorData};
use regex::Regex;
use std::error::Error;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    //read in text file for the verdict
    let contents = fs::read_to_string("verdict.txt")?;

    //regex split text into vector of tokens and characters
    let re = Regex::new(r#"([,.:;?_!"()\']|--|\s)"#).unwrap();

    // Split by regex and collect into Vec<String>
    let token_text: Vec<String> = re.split(&contents).map(|s| s.to_string()).collect();

    //encode vector using tiktoken
    let enc = tiktoken::get_encoding("o200k_base").unwrap();

    let tokens = enc.encode(&contents);

    //create input and target vectors x and y
    let context_size: usize = 256;
    let stride: usize = 128;

    let mut x_inputs = Vec::new();
    let mut y_targets = Vec::new();
    let mut i = 0;
    while i + context_size < tokens.len() {
        let input_chunk = tokens[i..i + context_size].to_vec();
        let target_chunk = tokens[i + 1..i + context_size + 1].to_vec();
        x_inputs.push(input_chunk);
        y_targets.push(target_chunk);
        i += stride;
    }

    //create the token embedding
    // Configure and initialize the embedding layer
    let vocab_size = 200019; //based on the o200k_base
    let output_dim = 768; //taken from GPT-2 small

    //define CPU backend
    let device = NdArrayDevice::Cpu;
    let embedding_layer: Embedding<NdArray> =
        EmbeddingConfig::new(vocab_size, output_dim).init(&device);
    // Convert the tokens vector into an i32 vec ->tensor for the Burn inputs
    // Convert to i32 — Burn's Int tensor uses i32 by default
    let tokens_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();

    // Wrap in a 2D tensor [batch_size=1, seq_len]
    let token_tensor: Tensor<NdArray, 2, Int> = Tensor::from_ints(
        TensorData::new(tokens_i32.clone(), Shape::new([1, tokens_i32.len()])),
        &device,
    );

    // Forward pass generates the continuous embeddings
    let embeddings: Tensor<NdArray, 3> = embedding_layer.forward(token_tensor);

    //generate positional embeddings
    let context_length: usize = 256;
    let output_dim: usize = 256;

    // Create the positional embedding layer
    let pos_embedding_layer: Embedding<NdArray> =
        EmbeddingConfig::new(context_length, output_dim).init(&device);

    // Equivalent of torch.arange(context_length) — positions [0, 1, 2, ..., 255]
    let positions: Vec<i32> = (0..context_length as i32).collect();
    let pos_tensor: Tensor<NdArray, 2, Int> = Tensor::from_ints(
        TensorData::new(positions, Shape::new([1, context_length])),
        &device,
    );

    // Forward pass — equivalent of pos_embedding_layer(torch.arange(context_length))
    let pos_embeddings = pos_embedding_layer.forward(pos_tensor);
    println!("{:?}", pos_embeddings);
    Ok(())
}
