use std::time::Instant;

use plotly::{
  common::{Line, TickMode},
  Plot, Scatter,
};
use stochastic_rs::{
  diffusions::ou::fou,
  jumps::jump_fou::jump_fou,
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

  for i in 0..1 {
    let d = jump_fou(0.1, 2.0, 0.5, 2.0, 2.0, 100, Some(0.0), Some(1.0));
    //let d = compound_poisson(50, 1.0, None, None, None);
    //let d = poisson(10.0, Some(50), None);

    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), d.clone())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new().color("blue"), //.shape(plotly::common::LineShape::Hv),
      )
      .name(format!("Sequence {}", i + 1).as_str());
    plot.add_trace(trace);
  }

  plot.show();

  println!("{}", start.elapsed().as_secs_f64());
}
