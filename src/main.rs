use std::time::Instant;

use plotly::{
  common::{Line, TickMode},
  Plot, Scatter,
};
use stochastic_rs::{
  diffusions::ou::fou,
  jumps::{jump_fou::jump_fou, merton::merton},
  noises::fgn::FgnFft,
  processes::{
    fbm::Fbm,
    poisson::{compound_poisson, poisson},
  },
  utils::Generator,
};

fn main() {
  let start = Instant::now();

  let mut plot = Plot::new();
  let fbm = Fbm::new(0.9, 1000, Some(1.0), Some(10));

  for i in 0..1 {
    let d = fbm.sample();
    let d = jump_fou(0.7, 30.0, 0.5, 1.0, 10.0, 500, Some(0.0), Some(10.0));

    let merton_path = merton(0.05, 0.2, 10.0, 1.0, 1000, Some(10.0), Some(5.0));
    //let [_, cp, _] = compound_poisson(None, 1.0, Some(10.), None, Some(0.2));
    //let d = poisson(10.0, Some(50), None);

    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), merton_path.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("blue")
          .shape(plotly::common::LineShape::Linear),
      )
      .name(format!("Sequence {}", i + 1).as_str());
    plot.add_trace(trace);
  }

  plot.show();

  println!("{}", start.elapsed().as_secs_f64());
}
