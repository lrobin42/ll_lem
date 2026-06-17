mod preprocessing;
use preprocessing::*;
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

    //create the token embeddings
    let embeddings = create_embeddings(&tokens);

    //add positional embeddings
    let pos_embeddings = add_positional_embeddings();

    Ok(())
}
