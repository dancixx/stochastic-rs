use std::time::Instant;

use plotly::{
  common::{Line, TickMode},
  Plot, Scatter,
};
use stochastic_rs::{
  diffusions::ou::fou,
  jumps::{bates::bates_1996, jump_fou::jump_fou, levy_diffusion::levy_diffusion, merton::merton},
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
    let jump_fou = jump_fou(0.7, 1.0, 0.5, 1.0, 10.0, 1000, Some(0.0), Some(1000.0));
    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), jump_fou.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("blue")
          .shape(plotly::common::LineShape::Linear),
      )
      .name("Jump FOU");
    //plot.add_trace(trace);
    let merton_path = merton(0.05, 0.2, 0.25, 1.0, 1000, Some(1000.0), Some(5.0));
    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), merton_path.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("red")
          .shape(plotly::common::LineShape::Hv),
      )
      .name("Merton");
    //plot.add_trace(trace);
    let levy_path = levy_diffusion(0.5, 1.0, 0.25, 1000, Some(0.0), Some(10.0));
    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), levy_path.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("green")
          .shape(plotly::common::LineShape::Linear),
      )
      .name("Levy");
    //plot.add_trace(trace);
    let [s, v] = bates_1996(
      0.05,
      1.5,
      0.04,
      0.3,
      -0.7,
      0.1,
      1000,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
    );

    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), s.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("black")
          .shape(plotly::common::LineShape::Linear),
      )
      .name("Bates 1996");
    //plot.add_trace(trace);

    let [_, cp, _] = compound_poisson(None, 1.0, Some(500.), None, None);
    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), cp.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("purple")
          .shape(plotly::common::LineShape::Hv),
      )
      .name("Compound Poisson");
    plot.add_trace(trace);
    let d = poisson(10.0, Some(50), None);
    let trace = Scatter::new((0..d.len()).collect::<Vec<_>>(), d.to_vec())
      .mode(plotly::common::Mode::Lines)
      .line(
        Line::new()
          .color("orange")
          .shape(plotly::common::LineShape::Hv),
      )
      .name("Poisson");
    //plot.add_trace(trace);
  }

  plot.show();

  println!("{}", start.elapsed().as_secs_f64());
}
