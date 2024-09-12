use stochastic_rs::{process::fbm::Fbm, Sampling};

fn main() {
  let fbm = Fbm::new(&Fbm {
    hurst: 0.9,
    n: 10000,
    t: None,
    m: Some(100),
    ..Default::default()
  });
  println!("{:?}", fbm.fgn.hurst);
  let mut runs = Vec::new();

  for _ in 0..20 {
    let start = std::time::Instant::now();
    for _ in 0..1000 {
      let _ = fbm.sample_par();
    }

    let duration = start.elapsed();
    println!(
      "Time elapsed in expensive_function() is: {:?}",
      duration.as_secs_f32()
    );
    runs.push(duration.as_secs_f32());
  }

  let sum: f32 = runs.iter().sum();
  let average = sum / runs.len() as f32;
  println!("Average time: {}", average);
}
