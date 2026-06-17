use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Int, Tensor, backend::Backend};
use burn::tensor::{Shape, TensorData};
mod preprocessing;
use preprocessing::*;
use regex::Regex;
use std::error::Error;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    //read in text file for the verdict by Edith Wharton
    let contents = fs::read_to_string("verdict.txt")?;

    //encode vector using tiktoken
    let enc = tiktoken::get_encoding("o200k_base").unwrap();
    let tokens = enc.encode(&contents);

    //create input and target vectors x and y
    let context_size: usize = 256;
    let stride: usize = 128;

    let (x_inputs, y_targets) = create_x_y_vectors(&tokens);

    //create the token embedding
    let embeddings = create_embeddings(&tokens);

    //generate positional embeddings
    // let context_length: usize = 256;
    // let output_dim: usize = 256;
    // let device = NdArrayDevice::Cpu;
    // // Create the positional embedding layer
    // let pos_embedding_layer: Embedding<NdArray> =
    //     EmbeddingConfig::new(context_length, output_dim).init(&device);

    // // Equivalent of torch.arange(context_length) — positions [0, 1, 2, ..., 255]
    // let positions: Vec<i32> = (0..context_length as i32).collect();
    // let pos_tensor: Tensor<NdArray, 2, Int> = Tensor::from_ints(
    //     TensorData::new(positions, Shape::new([1, context_length])),
    //     &device,
    // );

    // // Forward pass — equivalent of pos_embedding_layer(torch.arange(context_length))
    let pos_embeddings = add_positional_embeddings();
    // println!("{:?}", pos_embeddings);

    Ok(())
}
