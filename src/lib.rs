// https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/index.html

// // Increase the amount of detail Clippy searches for.
// #![warn(clippy::pedantic)]
// Strictly enforce documentation.
// #![forbid(missing_docs)]
// // When writing mathematical equations in documentation, Clippy suggests to
// // put backticks inside the LaTeX block. This suppresses that behavior.
// #![allow(clippy::doc_markdown)]

pub mod prelude;

pub mod diffusions;
pub mod jumps;
pub mod models;
pub mod noises;
pub mod processes;
pub mod statistics;
pub mod utils;
