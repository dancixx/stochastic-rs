use stochastic_rs::{noise::fgn::Fgn, Sampling};

fn main() {
  let fbm = Fgn::new(0.9, 10000, None, Some(10000));

  let start = std::time::Instant::now();
  for _ in 0..10000 {
    let _ = fbm.sample();
  }

  let duration = start.elapsed();
  println!(
    "Time elapsed in expensive_function() is: {:?}",
    duration.as_secs_f32()
  );

  let fbm = Fgn::new(0.9, 10000, None, Some(10000));

  let start = std::time::Instant::now();
  for _ in 0..10000 {
    let _ = fbm.sample();
  }

  let duration = start.elapsed();
  println!(
    "Time elapsed in expensive_function() is: {:?}",
    duration.as_secs_f32()
  );
}
