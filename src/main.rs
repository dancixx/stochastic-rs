use plotly::common::Mode;
use plotly::Plot;
use plotly::Scatter;
use std::time::Instant;

use stochastic_rs::{
  noises::fgn, processes::fbm, statistics::fractal_dim::higuchi_fd, utils::Generator,
};

fn main() {
  let m = 20;
  let hurst = 0.75;
  let n = 500;
  let fgn = fbm::Fbm::new(hurst, n, None, None);

  let start = Instant::now();
  let mut data_samples = Vec::new();

  for _ in 0..m {
    let data = fgn.sample();
    data_samples.push(data);
  }

  println!("elapsed {}", start.elapsed().as_secs_f64());

  // Create a Plotly plot
  let mut plot = Plot::new();

  for (i, data) in data_samples.iter().enumerate() {
    let trace = Scatter::new((0..data.len()).collect::<Vec<_>>(), data.clone())
      .mode(Mode::Lines)
      .name(format!("Sequence {}", i + 1).as_str());
    plot.add_trace(trace);
  }

  // Save the plot as an HTML file
  plot.write_html("plot.html");

  println!("Plot saved as plot.html");
}
