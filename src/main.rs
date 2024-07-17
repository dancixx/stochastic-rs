use std::time::Instant;

use stochastic_rs::{noises::fgn::FgnFft, utils::Generator};

fn main() {
  let start = Instant::now();

  let fgn = FgnFft::new(0.7, 1000, Some(1.0), None);

  for _ in 0..1000 {
    let d = fgn.sample();
    // println!("{:?}", d.len());
  }

  println!("{}", start.elapsed().as_secs_f64());
}
